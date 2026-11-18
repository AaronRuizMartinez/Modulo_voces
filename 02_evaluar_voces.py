#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_evaluar_voces.py
===================
Evaluación objetiva integral para las 3 voces híbridas sintéticas:
  - Voz_Hibrida_01 (vs. Donantes Voz_A_01 y Voz_B_01)
  - Voz_Hibrida_02 (vs. Donantes Voz_A_02 y Voz_B_02)
  - Voz_Hibrida_03 (vs. Donantes Voz_A_03 y Voz_B_03)

Métricas calculadas por cada una de las 3 voces (sobre sus 45 frases sintetizadas):
  1. Similitud de Hablante (Diferenciación de Donantes):
     - Similitud coseno media ± desviación estándar:
       * Voz Híbrida vs. Donante A
       * Voz Híbrida vs. Donante B
       * Donante A vs. Donante B
     - Soporta SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`) y `resemblyzer`.
  2. Estabilidad Inter-Frase:
     - Matriz de similitud coseno por pares (45 x 45 = 990 pares únicos fuera de la diagonal)
       calculando media, mínima, máxima y desviación estándar.
  3. Inteligibilidad (WER y CER):
     - Transcripción automática de los 45 audios x 3 voces (135 audios) con `faster-whisper`
       (`small` o `medium`, `language="es"`) y cálculo de WER y CER con `jiwer`
       normalizando puntuación y mayúsculas.

Exportaciones:
  - resultados/metricas_evaluacion.json
  - resultados/tabla_resumen_3_voces.csv
  - resultados/tabla_comparativa_3_voces.md
  - resultados/tabla_comparativa_3_voces.tex
"""

import argparse
import csv
import importlib.util
import json
import os
import re
import string
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.spatial.distance import cdist
from tqdm import tqdm

DEFAULT_VOICES = ["Voz_Hibrida_01", "Voz_Hibrida_02", "Voz_Hibrida_03"]
TARGET_SR = 22050

VOICE_DONOR_MAP: Dict[str, Tuple[str, str]] = {
    "Voz_Hibrida_01": ("Voz_A_01", "Voz_B_01"),
    "Voz_Hibrida_02": ("Voz_A_02", "Voz_B_02"),
    "Voz_Hibrida_03": ("Voz_A_03", "Voz_B_03"),
}


# ==============================================================================
# NORMALIZACIÓN DE TEXTO Y CÁLCULO DE WER / CER (CON JIWER)
# ==============================================================================
def normalizar_texto_es(texto: str) -> str:
    """
    Normaliza texto en español para evaluación ASR:
      - Convierte a minúsculas.
      - Elimina signos de puntuación (incluyendo ¿ ? ¡ ! , . ; : " ' -).
      - Colapsa espacios múltiples.
    """
    texto = texto.lower().strip()
    signos_extra = "¿?¡!«»“”‘’—–…" + string.punctuation
    tabla = str.maketrans("", "", signos_extra)
    texto = texto.translate(tabla)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def calcular_wer_cer(referencia: str, hipotesis: str) -> Tuple[float, float]:
    """
    Calcula Word Error Rate (WER) y Character Error Rate (CER) usando `jiwer`
    sobre textos normalizados en puntuación y mayúsculas.
    Incluye respaldo exacto por programación dinámica Levenshtein si jiwer no está instalado.
    """
    ref_norm = normalizar_texto_es(referencia)
    hyp_norm = normalizar_texto_es(hipotesis)

    if not ref_norm:
        return (0.0 if not hyp_norm else 1.0), (0.0 if not hyp_norm else 1.0)

    try:
        import jiwer

        wer_val = float(jiwer.wer(ref_norm, hyp_norm))
        cer_val = float(jiwer.cer(ref_norm, hyp_norm))
        return wer_val, cer_val
    except ImportError:
        # Fallback Levenshtein exacto a nivel de palabra y carácter
        def _levenshtein_rate(seq_ref: List[Any], seq_hyp: List[Any]) -> float:
            n, m = len(seq_ref), len(seq_hyp)
            dp = list(range(m + 1))
            for i in range(1, n + 1):
                prev, dp[0] = dp[0], i
                for j in range(1, m + 1):
                    temp = dp[j]
                    cost = 0 if seq_ref[i - 1] == seq_hyp[j - 1] else 1
                    dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                    prev = temp
            return float(dp[m]) / max(1, n)

        return (
            _levenshtein_rate(ref_norm.split(), hyp_norm.split()),
            _levenshtein_rate(list(ref_norm), list(hyp_norm)),
        )


# ==============================================================================
# EXTRACTOR DE SPEAKER EMBEDDINGS (SPEECHBRAIN / RESEMBLYZER / ACÚSTICO)
# ==============================================================================
class SpeakerEmbeddingExtractor:
    """
    Extractor de embeddings de hablante con soporte para:
      1. SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`)
      2. Resemblyzer (`VoiceEncoder`)
      3. Modo Acústico Determinista PyTorch/Librosa (para --dry-run o entorno offline)
    """

    def __init__(self, backend: str = "auto", device: Optional[str] = None, dry_run: bool = False) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dry_run = dry_run
        self.backend_requested = backend
        self.backend_active = "acoustic_torch_mfcc"
        self.model = None

        if self.dry_run and backend == "auto":
            print("[INFO] Modo --dry-run activo: usando extractor de embeddings espectrales PyTorch/Librosa.")
            return

        if backend in ("auto", "speechbrain"):
            if self._intentar_cargar_speechbrain():
                return
            if backend == "speechbrain":
                raise RuntimeError("No se pudo inicializar SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`).")

        if backend in ("auto", "resemblyzer"):
            if self._intentar_cargar_resemblyzer():
                return
            if backend == "resemblyzer":
                raise RuntimeError("No se pudo inicializar `resemblyzer`.")

        print("[WARN] Usando extractor de embeddings espectrales PyTorch/Librosa de respaldo.")

    def _intentar_cargar_speechbrain(self) -> bool:
        try:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier

            savedir = Path("modelos_base") / "speechbrain_ecapa"
            savedir.mkdir(parents=True, exist_ok=True)
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(savedir),
                run_opts={"device": self.device},
            )
            self.backend_active = "speechbrain/spkrec-ecapa-voxceleb"
            print(f"[INFO] Extractor de embeddings activo: {self.backend_active} ({self.device})")
            return True
        except Exception as exc:
            print(f"[WARN] SpeechBrain no disponible ({exc}). Intentando Resemblyzer...")
            return False

    def _intentar_cargar_resemblyzer(self) -> bool:
        try:
            from resemblyzer import VoiceEncoder

            self.model = VoiceEncoder(device=self.device)
            self.backend_active = "resemblyzer"
            print(f"[INFO] Extractor de embeddings activo: resemblyzer ({self.device})")
            return True
        except Exception as exc:
            print(f"[WARN] Resemblyzer no disponible ({exc}).")
            return False

    def extraer_embedding(self, wav_path: Path) -> np.ndarray:
        """Extrae un vector de embedding L2-normalizado a partir de un archivo WAV."""
        if not wav_path.exists():
            raise FileNotFoundError(f"Archivo de audio no encontrado: {wav_path}")

        if self.backend_active == "speechbrain/spkrec-ecapa-voxceleb":
            y, _ = librosa.load(str(wav_path), sr=16000, mono=True)
            tensor_wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_batch(tensor_wav).squeeze().detach().cpu().numpy()
        elif self.backend_active == "resemblyzer":
            from resemblyzer import preprocess_wav

            wav = preprocess_wav(str(wav_path))
            emb = self.model.embed_utterance(wav)
        else:
            # Vector de características acústicas de alta dimensión (MFCC + Delta + Spectral Contrast + Pitch/Formant stats)
            y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
            if len(y) < 1024:
                y = np.pad(y, (0, 1024 - len(y)))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            delta = librosa.feature.delta(mfcc)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

            feats = np.concatenate(
                [
                    np.mean(mfcc, axis=1),
                    np.std(mfcc, axis=1),
                    np.mean(delta, axis=1),
                    np.mean(contrast, axis=1),
                    np.std(contrast, axis=1),
                    [float(np.mean(centroid)) / 1000.0, float(np.mean(rolloff)) / 1000.0],
                ]
            ).astype(np.float32)
            emb = feats

        emb = np.asarray(emb, dtype=np.float32).flatten()
        norma = float(np.linalg.norm(emb))
        if norma > 1e-8:
            emb = emb / norma
        return emb


# ==============================================================================
# TRANSCRIPTOR ASR CON FASTER-WHISPER
# ==============================================================================
class TranscriptorInteligibilidad:
    """
    Wrapper sobre `faster-whisper` (modelo `small` o `medium`, `language="es"`)
    para calcular WER y CER en las 45 frases de cada una de las 3 voces.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        self.model_size = model_size
        self.dry_run = dry_run
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if compute_type:
            self.compute_type = compute_type
        else:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = None

        if not self.dry_run:
            self._cargar_modelo()

    def _cargar_modelo(self) -> None:
        from faster_whisper import WhisperModel

        print(
            f"[INFO] Cargando faster-whisper ({self.model_size}) en {self.device} "
            f"(compute_type={self.compute_type})..."
        )
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            print(f"[WARN] Falló carga en {self.device} ({exc}); reintentando en CPU (int8)...")
            self.device = "cpu"
            self.compute_type = "int8"
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

    def transcribir(self, wav_path: Path, texto_referencia: str, voice_name: str, phrase_idx: int) -> str:
        """Transcribe un archivo WAV en español usando faster-whisper (o simulación controlada en --dry-run)."""
        if self.dry_run:
            # En modo --dry-run los audios son ondas armónicas sintéticas sin fonemas reales.
            # Simulamos hipótesis realistas con pequeñas variaciones controladas para validar jiwer WER/CER.
            palabras = texto_referencia.split()
            rng = np.random.default_rng(abs(hash(f"asr_{voice_name}_{phrase_idx}")) % (2**32))
            if len(palabras) > 6 and rng.random() < 0.14:
                idx_mod = int(rng.integers(0, len(palabras)))
                palabra = palabras[idx_mod]
                if len(palabra) > 4:
                    palabras[idx_mod] = palabra[:-1]
            return " ".join(palabras)

        audio_16k, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        segments, _ = self.model.transcribe(
            audio_16k.astype(np.float32),
            language="es",
            beam_size=5,
            vad_filter=True,
        )
        texto_transcrito = " ".join(seg.text.strip() for seg in segments).strip()
        return texto_transcrito


# ==============================================================================
# UTILIDADES MATEMÁTICAS DE SIMILITUD COSENO
# ==============================================================================
def similitud_coseno(u: np.ndarray, v: np.ndarray) -> float:
    """Calcula la similitud coseno entre dos vectores L2-normalizados."""
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(u, v) / denom)


def estadisticas_matriz_inter_frase(embeddings: np.ndarray) -> Dict[str, Any]:
    """
    Calcula la matriz de similitud coseno (N x N) entre las N=45 frases de una voz
    y extrae media, desviación estándar, mínimo y máximo sobre los N*(N-1)/2 pares
    fuera de la diagonal principal.
    """
    # Normalizar filas
    normas = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normas = np.where(normas < 1e-9, 1.0, normas)
    emb_norm = embeddings / normas

    matriz_sim = np.clip(np.dot(emb_norm, emb_norm.T), -1.0, 1.0)
    n = matriz_sim.shape[0]
    if n < 2:
        return {
            "num_frases": n,
            "num_pares_comparados": 0,
            "media": 1.0,
            "std": 0.0,
            "min": 1.0,
            "max": 1.0,
            "matriz_resumen_5x5": matriz_sim.tolist(),
        }

    indices_triang_sup = np.triu_indices(n, k=1)
    valores_pares = matriz_sim[indices_triang_sup]

    return {
        "num_frases": int(n),
        "num_pares_comparados": int(len(valores_pares)),
        "media": float(np.mean(valores_pares)),
        "std": float(np.std(valores_pares)),
        "min": float(np.min(valores_pares)),
        "max": float(np.max(valores_pares)),
        "matriz_resumen_5x5": np.round(matriz_sim[:5, :5], 4).tolist(),
    }


# ==============================================================================
# GENERACIÓN DE TABLAS COMPARATIVAS (CSV, MARKDOWN Y LATEX)
# ==============================================================================
def exportar_tablas_comparativas(
    resumen_filas: List[Dict[str, Any]],
    resultados_dir: Path,
) -> Tuple[Path, Path, Path]:
    """
    Exporta las tablas comparativas consolidadas de las 3 voces en:
      - CSV: resultados/tabla_resumen_3_voces.csv
      - Markdown: resultados/tabla_comparativa_3_voces.md
      - LaTeX: resultados/tabla_comparativa_3_voces.tex
    """
    csv_path = resultados_dir / "tabla_resumen_3_voces.csv"
    md_path = resultados_dir / "tabla_comparativa_3_voces.md"
    tex_path = resultados_dir / "tabla_comparativa_3_voces.tex"

    columnas = [
        "voz",
        "donante_a",
        "donante_b",
        "num_frases",
        "sim_hibrida_vs_a_media",
        "sim_hibrida_vs_a_std",
        "sim_hibrida_vs_b_media",
        "sim_hibrida_vs_b_std",
        "sim_donante_a_vs_b",
        "estabilidad_inter_media",
        "estabilidad_inter_std",
        "estabilidad_inter_min",
        "estabilidad_inter_max",
        "wer_medio_pct",
        "wer_std_pct",
        "cer_medio_pct",
        "cer_std_pct",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(resumen_filas)

    # --- Tabla en Markdown ---
    lineas_md = [
        "# Tabla Comparativa Consolidada — Evaluación Objetiva de las 3 Voces Híbridas",
        "",
        "| Voz Híbrida | Par de Donantes | Sim. Híbrida vs A (Media ± SD) | Sim. Híbrida vs B (Media ± SD) | Sim. A vs B | Estabilidad Inter-Frase (Media ± SD [Mín]) | WER (%) | CER (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in resumen_filas:
        lineas_md.append(
            f"| **{r['voz']}** | `{r['donante_a']}` / `{r['donante_b']}` | "
            f"{r['sim_hibrida_vs_a_media']:.4f} ± {r['sim_hibrida_vs_a_std']:.4f} | "
            f"{r['sim_hibrida_vs_b_media']:.4f} ± {r['sim_hibrida_vs_b_std']:.4f} | "
            f"{r['sim_donante_a_vs_b']:.4f} | "
            f"{r['estabilidad_inter_media']:.4f} ± {r['estabilidad_inter_std']:.4f} "
            f"(mín: {r['estabilidad_inter_min']:.4f}) | "
            f"{r['wer_medio_pct']:.2f}% ± {r['wer_std_pct']:.2f}% | "
            f"{r['cer_medio_pct']:.2f}% ± {r['cer_std_pct']:.2f}% |"
        )
    lineas_md.append("")
    with open(md_path, "w", encoding="utf-8") as f_md:
        f_md.write("\n".join(lineas_md))

    # --- Tabla en LaTeX ---
    lineas_tex = [
        "% Tabla LaTeX generada automáticamente por 02_evaluar_voces.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Evaluación acústica objetiva de similitud de hablante, estabilidad inter-frase e inteligibilidad (WER/CER) para las 3 voces híbridas sintéticas ($N=45$ frases por voz).}",
        "\\label{tab:evaluacion_3_voces_hibridas}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\hline",
        "\\textbf{Voz Híbrida} & \\textbf{Donantes (A / B)} & "
        "\\textbf{Sim. Híb--A} & \\textbf{Sim. Híb--B} & \\textbf{Sim. A--B} & "
        "\\textbf{Estabilidad Inter-Frase} & \\textbf{WER (\\%)} & \\textbf{CER (\\%)} \\\\",
        "\\hline",
    ]
    for r in resumen_filas:
        voz_tex = r["voz"].replace("_", "\\_")
        dona_tex = f"{r['donante_a'].replace('_', '\\_')} / {r['donante_b'].replace('_', '\\_')}"
        lineas_tex.append(
            f"{voz_tex} & {dona_tex} & "
            f"${r['sim_hibrida_vs_a_media']:.3f} \\pm {r['sim_hibrida_vs_a_std']:.3f}$ & "
            f"${r['sim_hibrida_vs_b_media']:.3f} \\pm {r['sim_hibrida_vs_b_std']:.3f}$ & "
            f"${r['sim_donante_a_vs_b']:.3f}$ & "
            f"${r['estabilidad_inter_media']:.3f} \\pm {r['estabilidad_inter_std']:.3f}$ "
            f"(\\text{{mín }} ${r['estabilidad_inter_min']:.3f}$) & "
            f"${r['wer_medio_pct']:.2f} \\pm {r['wer_std_pct']:.2f}$ & "
            f"${r['cer_medio_pct']:.2f} \\pm {r['cer_std_pct']:.2f}$ \\\\"
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

    return csv_path, md_path, tex_path


# ==============================================================================
# PIPELINE PRINCIPAL DE EVALUACIÓN DE LAS 3 VOCES
# ==============================================================================
def asegurar_corpus_y_entradas_si_dry_run(base_dir: Path, voices: List[str], dry_run: bool) -> None:
    """
    Si se ejecuta 02_evaluar_voces.py directamente con --dry-run y aún no existen
    los audios de entrada o las 45 frases del corpus, invoca al módulo 01 para crearlos.
    """
    spec = importlib.util.spec_from_file_location(
        "mod01", str(base_dir / "01_generar_corpus_prueba.py")
    )
    if spec and spec.loader:
        mod01 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod01)
        mod01.crear_estructura_directorios(base_dir)
        mod01.importar_audios_existentes_workspace(base_dir)
        if dry_run:
            mod01.asegurar_audios_entrada_dummy(base_dir, voices, overwrite=False)
            # Verificar si faltan los 45 audios de alguna voz
            faltan_corpus = any(
                not (base_dir / "resultados" / "corpus_evaluacion" / v / "metadata.csv").exists()
                for v in voices
            )
            if faltan_corpus:
                print("[DRY-RUN] Generando corpus dummy previo desde 01_generar_corpus_prueba.py...")
                ns = argparse.Namespace(
                    voices=voices,
                    base_dir=str(base_dir),
                    api_url="http://localhost:8000/v1/audio/speech",
                    mode="auto",
                    model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    sample_rate=TARGET_SR,
                    num_phrases=45,
                    dry_run=True,
                )
                mod01.ejecutar_generacion_corpus(ns)


def cargar_metadata_voz(voz_corpus_dir: Path) -> List[Tuple[Path, str]]:
    """Lee metadata.csv en formato LJSpeech (nombre_archivo|texto_exacto)."""
    metadata_path = voz_corpus_dir / "metadata.csv"
    wavs_dir = voz_corpus_dir / "wavs"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No existe {metadata_path}. Ejecute primero 01_generar_corpus_prueba.py."
        )

    pares: List[Tuple[Path, str]] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or "|" not in linea:
                continue
            nombre_archivo, texto = linea.split("|", 1)
            nombre_archivo = nombre_archivo.strip()
            if not nombre_archivo.lower().endswith(".wav"):
                nombre_archivo = f"{nombre_archivo}.wav"
            pares.append((wavs_dir / nombre_archivo, texto.strip()))
    return pares


def ejecutar_evaluacion(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir).resolve()
    resultados_dir = base_dir / "resultados"
    donantes_dir = base_dir / "datos_entrada" / "donantes"
    hibridas_dir = base_dir / "datos_entrada" / "voces_hibridas"
    corpus_dir = resultados_dir / "corpus_evaluacion"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    asegurar_corpus_y_entradas_si_dry_run(base_dir, args.voices, args.dry_run)

    extractor = SpeakerEmbeddingExtractor(
        backend=args.embedding_backend,
        device=args.device,
        dry_run=args.dry_run,
    )
    transcriptor = TranscriptorInteligibilidad(
        model_size=args.whisper_model,
        device=args.device,
        dry_run=args.dry_run,
    )

    reporte_json: Dict[str, Any] = {
        "configuracion": {
            "voces_evaluadas": args.voices,
            "embedding_backend": extractor.backend_active,
            "whisper_model": args.whisper_model,
            "dry_run": args.dry_run,
        },
        "voces": {},
    }
    resumen_filas: List[Dict[str, Any]] = []

    print("\n" + "=" * 78)
    print(f"INICIANDO EVALUACIÓN OBJETIVA DE {len(args.voices)} VOCES HÍBRIDAS")
    print(f"Embeddings: {extractor.backend_active} | ASR: faster-whisper ({args.whisper_model})")
    print("=" * 78)

    for idx_v, voz in enumerate(args.voices, start=1):
        don_a, don_b = VOICE_DONOR_MAP.get(voz, (f"Voz_A_0{idx_v}", f"Voz_B_0{idx_v}"))
        ruta_don_a = donantes_dir / f"{don_a}.wav"
        ruta_don_b = donantes_dir / f"{don_b}.wav"
        ruta_hib_maestra = hibridas_dir / f"{voz}.wav"

        pares_corpus = cargar_metadata_voz(corpus_dir / voz)
        if not pares_corpus:
            raise RuntimeError(f"El corpus de {voz} está vacío.")

        # 1. Embeddings de los donantes y de la muestra híbrida maestra
        emb_don_a = extractor.extraer_embedding(ruta_don_a)
        emb_don_b = extractor.extraer_embedding(ruta_don_b)
        emb_hib_maestra = (
            extractor.extraer_embedding(ruta_hib_maestra)
            if ruta_hib_maestra.exists()
            else None
        )

        sim_a_vs_b = similitud_coseno(emb_don_a, emb_don_b)

        # 2. Recorrer las 45 frases para extraer embeddings y calcular WER/CER
        embeddings_frases: List[np.ndarray] = []
        sims_vs_a: List[float] = []
        sims_vs_b: List[float] = []
        sims_vs_maestra: List[float] = []
        wers_voz: List[float] = []
        cers_voz: List[float] = []
        detalle_frases: List[Dict[str, Any]] = []

        pbar = tqdm(
            enumerate(pares_corpus, start=1),
            total=len(pares_corpus),
            desc=f"Evaluando {voz}",
            unit="frase",
        )

        for idx_f, (wav_path, texto_original) in pbar:
            emb_frase = extractor.extraer_embedding(wav_path)
            embeddings_frases.append(emb_frase)

            s_a = similitud_coseno(emb_frase, emb_don_a)
            s_b = similitud_coseno(emb_frase, emb_don_b)
            sims_vs_a.append(s_a)
            sims_vs_b.append(s_b)

            s_m = (
                similitud_coseno(emb_frase, emb_hib_maestra)
                if emb_hib_maestra is not None
                else 1.0
            )
            sims_vs_maestra.append(s_m)

            hipotesis_asr = transcriptor.transcribir(
                wav_path=wav_path,
                texto_referencia=texto_original,
                voice_name=voz,
                phrase_idx=idx_f,
            )
            wer_f, cer_f = calcular_wer_cer(texto_original, hipotesis_asr)
            wers_voz.append(wer_f)
            cers_voz.append(cer_f)

            detalle_frases.append(
                {
                    "id_frase": idx_f,
                    "archivo": wav_path.name,
                    "texto_original": texto_original,
                    "transcripcion_asr": hipotesis_asr,
                    "sim_vs_donante_a": round(s_a, 4),
                    "sim_vs_donante_b": round(s_b, 4),
                    "sim_vs_hibrida_maestra": round(s_m, 4),
                    "wer": round(wer_f, 4),
                    "cer": round(cer_f, 4),
                }
            )
            pbar.set_postfix(
                {
                    "SimA": f"{s_a:.3f}",
                    "SimB": f"{s_b:.3f}",
                    "WER": f"{wer_f*100:.1f}%",
                }
            )

        # Si existe la muestra híbrida maestra, incluir también su similitud directa contra A y B
        sim_maestra_vs_a = (
            similitud_coseno(emb_hib_maestra, emb_don_a)
            if emb_hib_maestra is not None
            else float(np.mean(sims_vs_a))
        )
        sim_maestra_vs_b = (
            similitud_coseno(emb_hib_maestra, emb_don_b)
            if emb_hib_maestra is not None
            else float(np.mean(sims_vs_b))
        )

        # Métrica 2: Estabilidad Inter-Frase sobre la matriz 45x45
        matriz_emb = np.vstack(embeddings_frases)
        estabilidad = estadisticas_matriz_inter_frase(matriz_emb)

        # Estadísticas agregadas de Métrica 1 y Métrica 3
        sim_a_media = float(np.mean(sims_vs_a))
        sim_a_std = float(np.std(sims_vs_a))
        sim_b_media = float(np.mean(sims_vs_b))
        sim_b_std = float(np.std(sims_vs_b))

        wer_medio_pct = float(np.mean(wers_voz) * 100.0)
        wer_std_pct = float(np.std(wers_voz) * 100.0)
        cer_medio_pct = float(np.mean(cers_voz) * 100.0)
        cer_std_pct = float(np.std(cers_voz) * 100.0)

        reporte_json["voces"][voz] = {
            "donantes": {"donante_a": don_a, "donante_b": don_b},
            "metrica_1_similitud_hablante": {
                "hibrida_corpus_vs_donante_a": {"media": round(sim_a_media, 4), "std": round(sim_a_std, 4)},
                "hibrida_corpus_vs_donante_b": {"media": round(sim_b_media, 4), "std": round(sim_b_std, 4)},
                "hibrida_maestra_vs_donante_a": round(sim_maestra_vs_a, 4),
                "hibrida_maestra_vs_donante_b": round(sim_maestra_vs_b, 4),
                "donante_a_vs_donante_b": round(sim_a_vs_b, 4),
                "hibrida_corpus_vs_maestra": {
                    "media": round(float(np.mean(sims_vs_maestra)), 4),
                    "std": round(float(np.std(sims_vs_maestra)), 4),
                },
            },
            "metrica_2_estabilidad_inter_frase": {
                "num_frases": estabilidad["num_frases"],
                "num_pares_comparados": estabilidad["num_pares_comparados"],
                "similitud_media": round(estabilidad["media"], 4),
                "similitud_std": round(estabilidad["std"], 4),
                "similitud_min": round(estabilidad["min"], 4),
                "similitud_max": round(estabilidad["max"], 4),
                "muestra_matriz_5x5": estabilidad["matriz_resumen_5x5"],
            },
            "metrica_3_inteligibilidad_asr": {
                "modelo_asr": f"faster-whisper ({args.whisper_model})",
                "wer_medio": round(float(np.mean(wers_voz)), 4),
                "wer_std": round(float(np.std(wers_voz)), 4),
                "wer_medio_pct": round(wer_medio_pct, 2),
                "cer_medio": round(float(np.mean(cers_voz)), 4),
                "cer_std": round(float(np.std(cers_voz)), 4),
                "cer_medio_pct": round(cer_medio_pct, 2),
            },
            "detalle_45_frases": detalle_frases,
        }

        resumen_filas.append(
            {
                "voz": voz,
                "donante_a": don_a,
                "donante_b": don_b,
                "num_frases": len(pares_corpus),
                "sim_hibrida_vs_a_media": round(sim_a_media, 4),
                "sim_hibrida_vs_a_std": round(sim_a_std, 4),
                "sim_hibrida_vs_b_media": round(sim_b_media, 4),
                "sim_hibrida_vs_b_std": round(sim_b_std, 4),
                "sim_donante_a_vs_b": round(sim_a_vs_b, 4),
                "estabilidad_inter_media": round(estabilidad["media"], 4),
                "estabilidad_inter_std": round(estabilidad["std"], 4),
                "estabilidad_inter_min": round(estabilidad["min"], 4),
                "estabilidad_inter_max": round(estabilidad["max"], 4),
                "wer_medio_pct": round(wer_medio_pct, 2),
                "wer_std_pct": round(wer_std_pct, 2),
                "cer_medio_pct": round(cer_medio_pct, 2),
                "cer_std_pct": round(cer_std_pct, 2),
            }
        )

    # Guardar resultados/metricas_evaluacion.json
    json_path = resultados_dir / "metricas_evaluacion.json"
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(reporte_json, f_json, indent=2, ensure_ascii=False)

    csv_path, md_path, tex_path = exportar_tablas_comparativas(resumen_filas, resultados_dir)

    print("\n" + "=" * 78)
    print("RESULTADOS CONSOLIDADOS DE LAS 3 VOCES HÍBRIDAS:")
    print("=" * 78)
    print(Path(md_path).read_text(encoding="utf-8"))
    print("-" * 78)
    print(f"✓ JSON detallado exportado en : {json_path}")
    print(f"✓ Tabla CSV exportada en      : {csv_path}")
    print(f"✓ Tabla Markdown exportada en : {md_path}")
    print(f"✓ Tabla LaTeX exportada en    : {tex_path}")
    print("=" * 78 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="02 - Evaluación objetiva (Similitud, Estabilidad Inter-Frase y WER/CER) para las 3 voces híbridas."
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=DEFAULT_VOICES,
        help="Voces híbridas a evaluar (por defecto: Voz_Hibrida_01 Voz_Hibrida_02 Voz_Hibrida_03).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Directorio raíz del proyecto.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "speechbrain", "resemblyzer"],
        default="auto",
        help="Backend para extracción de speaker embeddings.",
    )
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="small",
        help="Tamaño del modelo faster-whisper para transcripción en español (por defecto: small).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de cómputo ('cuda' o 'cpu'). Por defecto autodetecta CUDA.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta validación rápida del flujo multi-voz con audios dummy y exportación completa de tablas.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ejecutar_evaluacion(parse_args())
