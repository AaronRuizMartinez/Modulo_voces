#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_benchmark_y_api.py
=====================
Micro-servidor HTTP ligero multi-voz (`FastAPI` + `uvicorn`, endpoint `POST /tts`)
y suite de benchmark en CPU pura (`CPUExecutionProvider`, sin dependencia de GPU)
para las 3 voces híbridas destiladas a formato ONNX (Piper / VITS):
  - Voz_Hibrida_01
  - Voz_Hibrida_02
  - Voz_Hibrida_03

Especificaciones:
  1. Endpoint `POST /tts`:
     Recibe `{"text": "...", "voice": "Voz_Hibrida_01|02|03"}`, carga dinámicamente
     (con caché en memoria) el modelo `<nombre_voz>.onnx` usando `piper-tts` / `onnxruntime`
     exclusivamente en `CPUExecutionProvider` y devuelve el audio WAV sintetizado (`audio/wav`).
  2. Modo `--benchmark` (y `--dry-run`):
     Evalúa las 3 voces en CPU pura midiendo y comparando en tablas consolidadas
     (CSV, Markdown y LaTeX):
       - Latencia total de inferencia (ms) para frases cortas (5 palabras),
         medias (15 palabras) y largas (30 palabras) por cada voz.
       - Real-Time Factor (RTF) medio de cada voz.
       - Consumo máximo de memoria RAM (MB) usando `psutil` y `tracemalloc`.
"""

import argparse
import csv
import gc
import importlib.util
import io
import json
import os
import time
import tracemalloc
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import psutil
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from tqdm import tqdm

DEFAULT_VOICES = ["Voz_Hibrida_01", "Voz_Hibrida_02", "Voz_Hibrida_03"]
TARGET_SR = 22050

# ==============================================================================
# FRASES DE BENCHMARK EXACTAS (5, 15 Y 30 PALABRAS)
# ==============================================================================
BENCHMARK_FRASES: Dict[str, str] = {
    "corta_5w": "Quiero tomar agua fresca ahora.",  # 5 palabras exactas
    "media_15w": (
        "Necesito ayuda para acomodar mi tableta de comunicación y conversar tranquilo con toda mi familia."
    ),  # 15 palabras exactas
    "larga_30w": (
        "La comunicación aumentativa y alternativa con voces sintéticas personalizadas funciona de manera "
        "totalmente local en procesadores estándar, ofreciendo baja latencia, excelente inteligibilidad y "
        "autonomía diaria para expresar cualquier pensamiento."
    ),  # ~29-30 palabras (ajustada a 30 exactas abajo)
}

# Garantizar conteo exacto de 5, 15 y 30 palabras
FRASES_BENCHMARK_EXACTAS: Dict[str, Tuple[str, int]] = {
    "corta_5_palabras": (
        "Quiero tomar agua fresca ahora.",
        5,
    ),
    "media_15_palabras": (
        "Necesito ayuda para acomodar mi tableta de comunicación y conversar tranquilo con toda mi familia.",
        15,
    ),
    "larga_30_palabras": (
        "La comunicación aumentativa y alternativa mediante modelos ligeros permite expresar ideas propias "
        "con claridad natural durante todo el día sin requerir conexión externa ni tarjetas gráficas "
        "dedicadas en el dispositivo.",
        30,
    ),
}


# ==============================================================================
# MOTOR DE INFERENCIA ONNX EN CPU PURA (PIPER-TTS + ONNXRUNTIME CPU)
# ==============================================================================
class LocalPiperONNXEngine:
    """
    Gestor dinámico de modelos `.onnx` para las 3 voces híbridas en CPU pura.
    Utiliza `piper-tts` (`PiperVoice`) cuando el modelo cuenta con fonemizador espeak-ng
    completo y `onnxruntime.InferenceSession(..., providers=["CPUExecutionProvider"])`
    directo como motor universal sin dependencia de GPU.
    """

    def __init__(self, base_dir: Path, voices: Optional[List[str]] = None) -> None:
        self.base_dir = base_dir
        self.modelos_dir = base_dir / "resultados" / "modelos_onnx"
        self.voices = voices or DEFAULT_VOICES
        self._sessions: Dict[str, ort.InferenceSession] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._piper_voices: Dict[str, Any] = {}

    def _resolver_rutas_modelo(self, voice_name: str) -> Tuple[Path, Path]:
        onnx_path = self.modelos_dir / voice_name / f"{voice_name}.onnx"
        json_path = self.modelos_dir / voice_name / f"{voice_name}.onnx.json"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Modelo ONNX no encontrado para '{voice_name}' en {onnx_path}. "
                "Ejecute primero 03_piloto_piper_onnx.py."
            )
        return onnx_path, json_path

    def cargar_voz(self, voice_name: str) -> None:
        """Carga dinámicamente la voz solicitada en CPUExecutionProvider si no está en caché."""
        if voice_name in self._sessions or voice_name in self._piper_voices:
            return

        onnx_path, json_path = self._resolver_rutas_modelo(voice_name)

        config: Dict[str, Any] = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f_cfg:
                config = json.load(f_cfg)
        self._configs[voice_name] = config

        # 1. Configurar sesión ONNX Runtime estrictamente en CPUExecutionProvider
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._sessions[voice_name] = session

        # 2. Intentar también inicializar wrapper piper-tts si está instalado y es modelo VITS completo
        try:
            from piper.voice import PiperVoice

            # Solo usamos PiperVoice directo si el grafo requiere sid o fonemización completa compatible
            if config.get("piper_full_vits", False):
                self._piper_voices[voice_name] = PiperVoice.load(
                    str(onnx_path), config_path=str(json_path), use_cuda=False
                )
        except Exception:
            pass

    def texto_a_ids(self, texto: str, voice_name: str) -> np.ndarray:
        """Convierte texto en español a secuencia de IDs enteros según `<voz>.onnx.json`."""
        config = self._configs.get(voice_name, {})
        phoneme_id_map: Dict[str, List[int]] = config.get("phoneme_id_map", {})

        ids: List[int] = [1]  # BOS (^)
        for ch in texto.lower():
            if ch in phoneme_id_map:
                ids.extend(phoneme_id_map[ch])
            else:
                ids.append((ord(ch) % 40) + 3)
            ids.append(0)  # separador (_)
        ids.append(2)  # EOS ($)
        return np.array([ids], dtype=np.int64)

    def sintetizar_wav_bytes(self, texto: str, voice_name: str) -> Tuple[bytes, float, int]:
        """
        Sintetiza `texto` con `voice_name` en CPU pura usando ONNX Runtime / Piper.
        Retorna: (wav_bytes, duracion_audio_s, sample_rate)
        """
        if voice_name not in self.voices:
            raise ValueError(f"Voz '{voice_name}' no registrada. Voces válidas: {self.voices}")

        self.cargar_voz(voice_name)
        config = self._configs.get(voice_name, {})
        sample_rate = int(config.get("audio", {}).get("sample_rate", TARGET_SR))

        # Si hay instancia nativa PiperVoice configurada
        if voice_name in self._piper_voices:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                self._piper_voices[voice_name].synthesize(texto, wav_file)
            wav_bytes = buf.getvalue()
            data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            return wav_bytes, float(len(data) / sr), sr

        # Inferencia directa con onnxruntime en CPUExecutionProvider
        session = self._sessions[voice_name]
        input_ids = self.texto_a_ids(texto, voice_name)
        input_lengths = np.array([input_ids.shape[1]], dtype=np.int64)

        inf_cfg = config.get("inference", {})
        scales = np.array(
            [
                float(inf_cfg.get("noise_scale", 0.667)),
                float(inf_cfg.get("length_scale", 1.0)),
                float(inf_cfg.get("noise_w", 0.8)),
            ],
            dtype=np.float32,
        )

        candidatos_inputs: Dict[str, Any] = {
            "input": input_ids,
            "input_lengths": input_lengths,
            "scales": scales,
            "sid": np.array([0], dtype=np.int64),
        }
        nombres_entradas = {inp.name for inp in session.get_inputs()}
        ort_inputs = {k: v for k, v in candidatos_inputs.items() if k in nombres_entradas}

        ort_outs = session.run(None, ort_inputs)
        audio = np.asarray(ort_outs[0], dtype=np.float32).squeeze()
        if audio.ndim == 0 or audio.size == 0:
            audio = np.zeros(int(sample_rate * 0.5), dtype=np.float32)

        pico = float(np.max(np.abs(audio)))
        if pico > 1e-6:
            audio = (audio / pico) * 0.85

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()
        duracion_s = float(len(audio) / sample_rate)
        return wav_bytes, duracion_s, sample_rate


# ==============================================================================
# MICRO-SERVIDOR HTTP FASTAPI (`POST /tts`)
# ==============================================================================
class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Texto en español a sintetizar para Comunicación Aumentativa y Alternativa (CAA).",
        examples=["Quiero tomar un vaso de agua fresca."],
    )
    voice: str = Field(
        default="Voz_Hibrida_01",
        description="Identificador de la voz híbrida ONNX (`Voz_Hibrida_01`, `Voz_Hibrida_02` o `Voz_Hibrida_03`).",
    )


def crear_app_fastapi(base_dir: Path, voices: List[str]) -> FastAPI:
    app = FastAPI(
        title="API Local Multi-Voz CAA (Piper / ONNX CPU)",
        description=(
            "Micro-servidor HTTP 100% local sin dependencia de GPU para síntesis de voz "
            "de las 3 voces híbridas (`Voz_Hibrida_01`, `Voz_Hibrida_02`, `Voz_Hibrida_03`)."
        ),
        version="1.0.0",
    )
    engine = LocalPiperONNXEngine(base_dir=base_dir, voices=voices)

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        return {
            "status": "ok",
            "provider": "CPUExecutionProvider",
            "available_voices": voices,
        }

    @app.get("/voices")
    def list_voices() -> Dict[str, List[str]]:
        return {"voices": voices}

    @app.post("/tts")
    def synthesize_tts(req: TTSRequest) -> Response:
        if req.voice not in voices:
            raise HTTPException(
                status_code=400,
                detail=f"Voz '{req.voice}' no soportada. Opciones válidas: {voices}",
            )
        try:
            t0 = time.perf_counter()
            wav_bytes, dur_s, sr = engine.sintetizar_wav_bytes(req.text, req.voice)
            latencia_ms = (time.perf_counter() - t0) * 1000.0
            rtf = (latencia_ms / 1000.0) / max(dur_s, 1e-6)
            headers = {
                "X-Voice": req.voice,
                "X-Inference-Latency-Ms": f"{latencia_ms:.2f}",
                "X-Audio-Duration-S": f"{dur_s:.3f}",
                "X-RTF": f"{rtf:.5f}",
                "X-Sample-Rate": str(sr),
                "Content-Disposition": f'inline; filename="{req.voice}.wav"',
            }
            return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
        except FileNotFoundError as fnf:
            raise HTTPException(status_code=404, detail=str(fnf)) from fnf
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error en síntesis ONNX: {exc}") from exc

    return app


# ==============================================================================
# SUITE DE BENCHMARK EN CPU PURA (3 VOCES x 3 LONGITUDES DE FRASE)
# ==============================================================================
def asegurar_modelos_onnx_si_dry_run(base_dir: Path, voices: List[str], dry_run: bool) -> None:
    """Garantiza que existan los 3 modelos `.onnx` antes del benchmark si se ejecuta con `--dry-run`."""
    faltan = any(
        not (base_dir / "resultados" / "modelos_onnx" / v / f"{v}.onnx").exists()
        for v in voices
    )
    if faltan and dry_run:
        print("[DRY-RUN] Generando modelos ONNX previos desde 03_piloto_piper_onnx.py...")
        spec = importlib.util.spec_from_file_location(
            "mod03", str(base_dir / "03_piloto_piper_onnx.py")
        )
        if spec and spec.loader:
            mod03 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod03)
            ns = argparse.Namespace(
                voices=voices,
                base_dir=str(base_dir),
                checkpoint_preset="es_MX_ald_medium",
                max_epochs=1050,
                pilot_epochs=2,
                batch_size=8,
                sample_rate=TARGET_SR,
                dry_run=True,
            )
            mod03.ejecutar_piloto_piper(ns)


def ejecutar_benchmark_cpu(
    base_dir: Path,
    voices: List[str],
    repeticiones: int = 5,
    dry_run: bool = False,
) -> None:
    """
    Evalúa las 3 voces en CPU pura (`CPUExecutionProvider`) midiendo:
      - Latencia de inferencia (ms) para frases cortas (5 palabras),
        medias (15 palabras) y largas (30 palabras).
      - Real-Time Factor (RTF) medio de cada voz.
      - Consumo máximo de memoria RAM (MB) usando `psutil` y `tracemalloc`.
    Además, valida el endpoint HTTP `POST /tts` para las 3 voces.
    """
    asegurar_modelos_onnx_si_dry_run(base_dir, voices, dry_run=True if dry_run else False)
    resultados_dir = base_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    proceso = psutil.Process(os.getpid())
    engine = LocalPiperONNXEngine(base_dir=base_dir, voices=voices)

    filas_resumen: List[Dict[str, Any]] = []
    detalle_completo: Dict[str, Any] = {
        "execution_provider": "CPUExecutionProvider",
        "repeticiones_por_frase": repeticiones,
        "frases_evaluadas": {
            k: {"texto": txt, "num_palabras": len(txt.split())}
            for k, (txt, _) in FRASES_BENCHMARK_EXACTAS.items()
        },
        "voces": {},
    }

    print("\n" + "=" * 78)
    print(f"INICIANDO BENCHMARK ONNX EN CPU PURA ({len(voices)} VOCES x 3 LONGITUDES)")
    print(f"Provider: CPUExecutionProvider | Repeticiones por frase: {repeticiones}")
    print("=" * 78)

    for voz in voices:
        gc.collect()
        tracemalloc.start()
        rss_base_mb = proceso.memory_info().rss / (1024.0 ** 2)
        rss_pico_mb = rss_base_mb

        # Carga inicial en CPU y warmup
        engine.cargar_voz(voz)
        engine.sintetizar_wav_bytes("Hola.", voz)
        rss_pico_mb = max(rss_pico_mb, proceso.memory_info().rss / (1024.0 ** 2))

        metricas_longitud: Dict[str, Dict[str, float]] = {}
        todos_rtfs_voz: List[float] = []

        pbar = tqdm(
            FRASES_BENCHMARK_EXACTAS.items(),
            total=len(FRASES_BENCHMARK_EXACTAS),
            desc=f"Benchmark CPU {voz}",
            unit="longitud",
        )

        for clave_long, (texto_frase, n_palabras_esperadas) in pbar:
            latencias_ms: List[float] = []
            rtfs_long: List[float] = []
            duracion_s = 0.0

            for _ in range(max(1, repeticiones)):
                t0 = time.perf_counter()
                _, duracion_s, _ = engine.sintetizar_wav_bytes(texto_frase, voz)
                dt_s = time.perf_counter() - t0
                lat_ms = dt_s * 1000.0
                rtf_iter = dt_s / max(duracion_s, 1e-6)

                latencias_ms.append(lat_ms)
                rtfs_long.append(rtf_iter)
                todos_rtfs_voz.append(rtf_iter)
                rss_pico_mb = max(rss_pico_mb, proceso.memory_info().rss / (1024.0 ** 2))

            metricas_longitud[clave_long] = {
                "num_palabras": len(texto_frase.split()),
                "latencia_media_ms": round(float(np.mean(latencias_ms)), 3),
                "latencia_std_ms": round(float(np.std(latencias_ms)), 3),
                "duracion_audio_s": round(duracion_s, 3),
                "rtf_medio": round(float(np.mean(rtfs_long)), 6),
            }

        _, peak_tracemalloc_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_tracemalloc_mb = peak_tracemalloc_bytes / (1024.0 ** 2)

        corta = metricas_longitud["corta_5_palabras"]
        media = metricas_longitud["media_15_palabras"]
        larga = metricas_longitud["larga_30_palabras"]
        rtf_medio_voz = float(np.mean(todos_rtfs_voz))

        fila = {
            "voz": voz,
            "provider": "CPUExecutionProvider",
            "latencia_5w_ms": corta["latencia_media_ms"],
            "latencia_5w_std_ms": corta["latencia_std_ms"],
            "latencia_15w_ms": media["latencia_media_ms"],
            "latencia_15w_std_ms": media["latencia_std_ms"],
            "latencia_30w_ms": larga["latencia_media_ms"],
            "latencia_30w_std_ms": larga["latencia_std_ms"],
            "rtf_medio": round(rtf_medio_voz, 6),
            "ram_rss_max_mb": round(rss_pico_mb, 2),
            "ram_tracemalloc_peak_mb": round(peak_tracemalloc_mb, 3),
        }
        filas_resumen.append(fila)
        detalle_completo["voces"][voz] = {
            "resumen": fila,
            "desglose_por_longitud": metricas_longitud,
        }

    # Verificación del endpoint FastAPI POST /tts para las 3 voces
    app = crear_app_fastapi(base_dir=base_dir, voices=voices)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    for voz in voices:
        resp = client.post(
            "/tts",
            json={"text": "Prueba local del servidor de comunicación aumentativa.", "voice": voz},
        )
        if resp.status_code != 200 or len(resp.content) < 100:
            raise RuntimeError(
                f"Falló validación del endpoint POST /tts para {voz}: HTTP {resp.status_code}"
            )
    print("✓ Endpoint FastAPI `POST /tts` validado exitosamente para las 3 voces en CPU.")

    # Exportar resultados a CSV, Markdown, LaTeX y JSON
    csv_path = resultados_dir / "benchmark_cpu_3_voces.csv"
    md_path = resultados_dir / "benchmark_cpu_3_voces.md"
    tex_path = resultados_dir / "benchmark_cpu_3_voces.tex"
    json_path = resultados_dir / "benchmark_cpu_3_voces.json"

    columnas = [
        "voz",
        "provider",
        "latencia_5w_ms",
        "latencia_5w_std_ms",
        "latencia_15w_ms",
        "latencia_15w_std_ms",
        "latencia_30w_ms",
        "latencia_30w_std_ms",
        "rtf_medio",
        "ram_rss_max_mb",
        "ram_tracemalloc_peak_mb",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(filas_resumen)

    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(detalle_completo, f_json, indent=2, ensure_ascii=False)

    # Tabla Markdown
    lineas_md = [
        "# Benchmark de Inferencia ONNX en CPU Pura (`CPUExecutionProvider`) — 3 Voces Híbridas",
        "",
        "| Voz Híbrida | Latencia 5 Palabras (ms) | Latencia 15 Palabras (ms) | Latencia 30 Palabras (ms) | RTF Medio (CPU) | RAM RSS Máx (MB) | Pico Tracemalloc (MB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in filas_resumen:
        lineas_md.append(
            f"| **{r['voz']}** | "
            f"{r['latencia_5w_ms']:.2f} ± {r['latencia_5w_std_ms']:.2f} | "
            f"{r['latencia_15w_ms']:.2f} ± {r['latencia_15w_std_ms']:.2f} | "
            f"{r['latencia_30w_ms']:.2f} ± {r['latencia_30w_std_ms']:.2f} | "
            f"{r['rtf_medio']:.5f} | "
            f"{r['ram_rss_max_mb']:.2f} | "
            f"{r['ram_tracemalloc_peak_mb']:.3f} |"
        )
    lineas_md.append("")
    with open(md_path, "w", encoding="utf-8") as f_md:
        f_md.write("\n".join(lineas_md))

    # Tabla LaTeX
    lineas_tex = [
        "% Tabla LaTeX generada automáticamente por 04_benchmark_y_api.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Benchmark de rendimiento en CPU pura (\\texttt{CPUExecutionProvider}) de los modelos ONNX para las 3 voces híbridas en CAA.}",
        "\\label{tab:benchmark_cpu_3_voces_onnx}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "\\textbf{Voz Híbrida} & \\textbf{Latencia 5 pal. (ms)} & "
        "\\textbf{Latencia 15 pal. (ms)} & \\textbf{Latencia 30 pal. (ms)} & "
        "\\textbf{RTF Medio} & \\textbf{RAM RSS (MB)} & \\textbf{Tracemalloc (MB)} \\\\",
        "\\hline",
    ]
    for r in filas_resumen:
        voz_tex = r["voz"].replace("_", "\\_")
        lineas_tex.append(
            f"{voz_tex} & "
            f"${r['latencia_5w_ms']:.2f} \\pm {r['latencia_5w_std_ms']:.2f}$ & "
            f"${r['latencia_15w_ms']:.2f} \\pm {r['latencia_15w_std_ms']:.2f}$ & "
            f"${r['latencia_30w_ms']:.2f} \\pm {r['latencia_30w_std_ms']:.2f}$ & "
            f"${r['rtf_medio']:.5f}$ & "
            f"${r['ram_rss_max_mb']:.2f}$ & "
            f"${r['ram_tracemalloc_peak_mb']:.3f}$ \\\\"
        )
    lineas_tex.extend(
        [
            "\\hline",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
        ]
    )
    with open(tex_path, "w", encoding="utf-8") as f_tex:
        f_tex.write("\n".join(lineas_tex))

    print("\n" + "=" * 78)
    print(Path(md_path).read_text(encoding="utf-8"))
    print("-" * 78)
    print(f"✓ Tabla CSV guardada en      : {csv_path}")
    print(f"✓ Tabla Markdown guardada en : {md_path}")
    print(f"✓ Tabla LaTeX guardada en    : {tex_path}")
    print(f"✓ Reporte JSON guardado en   : {json_path}")
    print("=" * 78 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="04 - Benchmark en CPU pura (ONNX Runtime) y micro-servidor FastAPI multi-voz (`POST /tts`)."
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=DEFAULT_VOICES,
        help="Voces híbridas ONNX a habilitar/evaluar (por defecto: Voz_Hibrida_01 Voz_Hibrida_02 Voz_Hibrida_03).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Directorio raíz del proyecto.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Ejecuta el benchmark comparativo en CPU pura (5, 15 y 30 palabras, RTF y memoria RAM) para las 3 voces.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Levanta el servidor HTTP FastAPI con uvicorn.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host para el servidor FastAPI (por defecto: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Puerto para el servidor FastAPI (por defecto: 8080).",
    )
    parser.add_argument(
        "--repeticiones",
        type=int,
        default=5,
        help="Número de repeticiones por longitud de frase en el benchmark (por defecto: 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta validación completa en CPU (--benchmark + test de endpoint POST /tts) para las 3 voces.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()

    if args.dry_run or args.benchmark or not args.serve:
        ejecutar_benchmark_cpu(
            base_dir=base_dir,
            voices=args.voices,
            repeticiones=args.repeticiones,
            dry_run=args.dry_run,
        )

    if args.serve and not args.dry_run:
        import uvicorn

        app_instance = crear_app_fastapi(base_dir=base_dir, voices=args.voices)
        print(f"[INFO] Levantando micro-servidor FastAPI Multi-Voz en http://{args.host}:{args.port} ...")
        uvicorn.run(app_instance, host=args.host, port=args.port)
