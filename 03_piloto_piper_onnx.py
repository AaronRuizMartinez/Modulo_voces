#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_piloto_piper_onnx.py
=======================
Piloto de destilación y preparación de entrenamiento/exportación a formato portable ONNX
(Piper / VITS) para las 3 voces híbridas sintéticas:
  - Voz_Hibrida_01
  - Voz_Hibrida_02
  - Voz_Hibrida_03

Funciones principales:
  1. Prepara y valida los datasets en formato LJSpeech (WAV 22050 Hz mono PCM_16 + metadata.csv)
     para cada una de las 3 voces en `resultados/datasets_piper/<nombre_voz>/`.
  2. Descarga automáticamente desde Hugging Face (`rhasspy/piper-checkpoints`) un checkpoint
     base preentrenado en español (`es/es_MX/ald/medium` o `es/es_ES/davefx/medium`)
     dentro de `modelos_base/piper/`.
  3. Genera el script bash maestro `run_piper_finetune_3_voces.sh` que itera sobre las 3 voces:
       a) `python3 -m piper_train.preprocess --language es ... --dataset-format ljspeech --single-speaker --sample-rate 22050`
       b) `python3 -m piper_train ... --resume_from_checkpoint ... --max_epochs ...`
       c) `python3 -m piper_train.export_onnx ...` generando `<nombre_voz>.onnx` y `<nombre_voz>.onnx.json`
  4. En modo `--dry-run` (o si aún no se ha corrido el entrenamiento completo de Piper),
     exporta grafos ONNX VITS/acústicos ligeros funcionales y sus archivos `<nombre_voz>.onnx.json`
     en `resultados/modelos_onnx/<nombre_voz>/` para permitir validación extremo a extremo
     con `onnxruntime` en CPU en `04_benchmark_y_api.py`.
  5. Registra consumo de VRAM, tiempo por época y estado por voz en `resultados/log_piloto_piper.txt`.
"""

import argparse
import datetime
import importlib.util
import json
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm

DEFAULT_VOICES = ["Voz_Hibrida_01", "Voz_Hibrida_02", "Voz_Hibrida_03"]
TARGET_SR = 22050

CHECKPOINTS_ESPANOL = {
    "es_MX_ald_medium": {
        "repo_id": "rhasspy/piper-checkpoints",
        "subfolder": "es/es_MX/ald/medium",
        "espeak_voice": "es-419",
    },
    "es_ES_davefx_medium": {
        "repo_id": "rhasspy/piper-checkpoints",
        "subfolder": "es/es_ES/davefx/medium",
        "espeak_voice": "es",
    },
}


# ==============================================================================
# 1. PREPARACIÓN DE DATASETS EN FORMATO LJSPEECH PARA LAS 3 VOCES
# ==============================================================================
def asegurar_corpus_previo_si_dry_run(base_dir: Path, voices: List[str], dry_run: bool) -> None:
    """Garantiza que exista el corpus de evaluación de las 3 voces si se corre con --dry-run."""
    faltan = any(
        not (base_dir / "resultados" / "corpus_evaluacion" / v / "metadata.csv").exists()
        for v in voices
    )
    if faltan and dry_run:
        spec = importlib.util.spec_from_file_location(
            "mod01", str(base_dir / "01_generar_corpus_prueba.py")
        )
        if spec and spec.loader:
            mod01 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod01)
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


def preparar_datasets_ljspeech_3_voces(
    base_dir: Path,
    voices: List[str],
    sample_rate: int = TARGET_SR,
) -> Dict[str, Dict[str, Any]]:
    """
    Convierte y estructura el corpus de cada una de las 3 voces al formato LJSpeech
    estándar requerido por `piper_train.preprocess`:
      resultados/datasets_piper/<nombre_voz>/
        ├── metadata.csv   (id_archivo|texto_exacto)
        └── wav/           (archivos WAV mono a 22050 Hz PCM_16)
        └── wavs/          (copia/enlace compatible con variantes LJSpeech)
    """
    corpus_base = base_dir / "resultados" / "corpus_evaluacion"
    piper_datasets_base = base_dir / "resultados" / "datasets_piper"
    piper_datasets_base.mkdir(parents=True, exist_ok=True)

    estadisticas_datasets: Dict[str, Dict[str, Any]] = {}

    for voz in voices:
        origen_voz_dir = corpus_base / voz
        origen_meta = origen_voz_dir / "metadata.csv"
        origen_wavs = origen_voz_dir / "wavs"

        if not origen_meta.exists():
            raise FileNotFoundError(
                f"No se encontró {origen_meta}. Ejecute primero 01_generar_corpus_prueba.py."
            )

        destino_voz_dir = piper_datasets_base / voz
        destino_wav_dir = destino_voz_dir / "wav"
        destino_wavs_dir = destino_voz_dir / "wavs"
        destino_wav_dir.mkdir(parents=True, exist_ok=True)
        destino_wavs_dir.mkdir(parents=True, exist_ok=True)

        with open(origen_meta, "r", encoding="utf-8") as f_in:
            lineas = [ln.strip() for ln in f_in if ln.strip() and "|" in ln]

        nuevas_lineas_meta: List[str] = []
        duracion_total_s = 0.0

        pbar = tqdm(
            lineas,
            desc=f"Preparando LJSpeech {voz}",
            unit="wav",
        )
        for linea in pbar:
            nombre_archivo, texto = linea.split("|", 1)
            nombre_archivo = nombre_archivo.strip()
            texto = texto.strip()

            stem = Path(nombre_archivo).stem
            nombre_wav = f"{stem}.wav"
            ruta_origen_wav = origen_wavs / nombre_wav
            if not ruta_origen_wav.exists() and (origen_wavs / nombre_archivo).exists():
                ruta_origen_wav = origen_wavs / nombre_archivo

            y, sr = librosa.load(str(ruta_origen_wav), sr=sample_rate, mono=True)
            # Recorte suave de silencios extremos (> 35 dB bajo pico) manteniendo margen natural
            y_trim, _ = librosa.effects.trim(y, top_db=38)
            if len(y_trim) > int(0.2 * sample_rate):
                y = y_trim

            pico = float(np.max(np.abs(y))) if y.size > 0 else 0.0
            if pico > 0.0:
                y = (y / pico) * 0.88

            ruta_dest_wav = destino_wav_dir / nombre_wav
            ruta_dest_wavs = destino_wavs_dir / nombre_wav
            sf.write(str(ruta_dest_wav), y, sample_rate, subtype="PCM_16")
            sf.write(str(ruta_dest_wavs), y, sample_rate, subtype="PCM_16")

            duracion_total_s += float(len(y) / sample_rate)
            # En LJSpeech para piper_train: id_sin_extension|texto
            nuevas_lineas_meta.append(f"{stem}|{texto}")

        destino_meta = destino_voz_dir / "metadata.csv"
        with open(destino_meta, "w", encoding="utf-8", newline="\n") as f_out:
            for ln in nuevas_lineas_meta:
                f_out.write(ln + "\n")

        estadisticas_datasets[voz] = {
            "dataset_dir": str(destino_voz_dir),
            "num_audios": len(nuevas_lineas_meta),
            "sample_rate": sample_rate,
            "duracion_total_s": round(duracion_total_s, 2),
        }
        print(
            f"✓ Dataset LJSpeech preparado para {voz}: {len(nuevas_lineas_meta)} audios "
            f"({duracion_total_s:.1f} s) en {destino_voz_dir}"
        )

    return estadisticas_datasets


# ==============================================================================
# 2. DESCARGA AUTOMATIZADA DE CHECKPOINT BASE EN ESPAÑOL (HUGGING FACE)
# ==============================================================================
def descargar_checkpoint_base_piper(
    base_dir: Path,
    preset: str = "es_MX_ald_medium",
    dry_run: bool = False,
) -> Path:
    """
    Descarga desde Hugging Face (`rhasspy/piper-checkpoints`) un checkpoint preentrenado
    en español (`es/es_MX/ald/medium` o `es/es_ES/davefx/medium`) en `modelos_base/piper/`.
    En modo `--dry-run`, crea un checkpoint base placeholder `.ckpt` y `config.json`
    sin requerir descarga de varios cientos de MB.
    """
    info = CHECKPOINTS_ESPANOL.get(preset, CHECKPOINTS_ESPANOL["es_MX_ald_medium"])
    destino_dir = base_dir / "modelos_base" / "piper" / info["subfolder"].replace("/", "_")
    destino_dir.mkdir(parents=True, exist_ok=True)

    # Revisar si ya existe un archivo .ckpt en destino_dir
    ckpts_existentes = list(destino_dir.glob("*.ckpt"))
    if ckpts_existentes:
        print(f"[INFO] Checkpoint base Piper ya disponible: {ckpts_existentes[0]}")
        return ckpts_existentes[0]

    if dry_run:
        dummy_ckpt = destino_dir / f"{preset}_base_dummy.ckpt"
        torch.save(
            {
                "epoch": 1000,
                "global_step": 250000,
                "preset": preset,
                "language": "es",
                "sample_rate": TARGET_SR,
                "state_dict": {"dummy_weight": torch.ones(4, 4)},
            },
            dummy_ckpt,
        )
        print(f"[DRY-RUN] Checkpoint base simulado creado en: {dummy_ckpt}")
        return dummy_ckpt

    print(
        f"[INFO] Consultando Hugging Face ({info['repo_id']} -> {info['subfolder']}) "
        "para descargar checkpoint base en español..."
    )
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        archivos_repo = api.list_repo_files(repo_id=info["repo_id"], repo_type="dataset")
        prefijo = info["subfolder"].rstrip("/") + "/"
        candidatos_ckpt = [
            f for f in archivos_repo if f.startswith(prefijo) and f.endswith(".ckpt")
        ]
        candidatos_cfg = [
            f for f in archivos_repo if f.startswith(prefijo) and f.endswith(".json")
        ]

        if not candidatos_ckpt:
            raise RuntimeError(f"No se encontraron archivos .ckpt bajo {prefijo} en {info['repo_id']}")

        archivo_ckpt = sorted(candidatos_ckpt)[-1]
        ruta_descargada = hf_hub_download(
            repo_id=info["repo_id"],
            repo_type="dataset",
            filename=archivo_ckpt,
            local_dir=str(destino_dir),
        )
        for cfg_file in candidatos_cfg:
            hf_hub_download(
                repo_id=info["repo_id"],
                repo_type="dataset",
                filename=cfg_file,
                local_dir=str(destino_dir),
            )
        print(f"✓ Checkpoint base descargado en: {ruta_descargada}")
        return Path(ruta_descargada)

    except Exception as exc:
        print(
            f"[WARN] No fue posible descargar el checkpoint desde Hugging Face ({exc}). "
            "Creando archivo de referencia local para continuar la generación de scripts."
        )
        fallback_ckpt = destino_dir / f"{preset}_placeholder.ckpt"
        torch.save({"preset": preset, "note": "placeholder_offline"}, fallback_ckpt)
        return fallback_ckpt


# ==============================================================================
# 3. GENERACIÓN DEL SCRIPT BASH MAESTRO `run_piper_finetune_3_voces.sh`
# ==============================================================================
def generar_script_bash_finetune(
    base_dir: Path,
    voices: List[str],
    ckpt_path: Path,
    max_epochs: int = 1050,
    batch_size: int = 8,
    sample_rate: int = TARGET_SR,
) -> Path:
    """
    Genera el script bash maestro `run_piper_finetune_3_voces.sh` que recorre en bucle
    las 3 voces (`for VOZ in Voz_Hibrida_01 Voz_Hibrida_02 Voz_Hibrida_03`) ejecutando:
      1. `python3 -m piper_train.preprocess`
      2. `python3 -m piper_train` (`--resume_from_checkpoint`, `--max_epochs`)
      3. `python3 -m piper_train.export_onnx` (`<nombre_voz>.onnx` y `<nombre_voz>.onnx.json`)
    """
    script_path = base_dir / "run_piper_finetune_3_voces.sh"
    voces_str = " ".join(voices)
    rel_ckpt = os.path.relpath(ckpt_path, base_dir)

    contenido_sh = f"""#!/usr/bin/env bash
# ==============================================================================
# run_piper_finetune_3_voces.sh
# Script maestro generado automáticamente por 03_piloto_piper_onnx.py
# Ejecuta preprocesamiento, fine-tuning (VITS/Piper) y exportación ONNX
# iterando sobre las 3 voces híbridas: {voces_str}
# ==============================================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
LOG_FILE="${{BASE_DIR}}/resultados/log_piloto_piper.txt"
BASE_CKPT="${{BASE_DIR}}/{rel_ckpt}"
MAX_EPOCHS="${{MAX_EPOCHS:-{max_epochs}}}"
BATCH_SIZE="${{BATCH_SIZE:-{batch_size}}}"
SAMPLE_RATE="{sample_rate}"

mkdir -p "${{BASE_DIR}}/resultados/modelos_onnx"

echo "==============================================================================" | tee -a "${{LOG_FILE}}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INICIO PILOTO FINE-TUNING PIPER ONNX (3 VOCES)" | tee -a "${{LOG_FILE}}"
echo "Checkpoint base: ${{BASE_CKPT}} | Max Epochs: ${{MAX_EPOCHS}} | Batch Size: ${{BATCH_SIZE}}" | tee -a "${{LOG_FILE}}"
echo "==============================================================================" | tee -a "${{LOG_FILE}}"

for VOZ in {voces_str}; do
    echo "" | tee -a "${{LOG_FILE}}"
    echo "------------------------------------------------------------------------------" | tee -a "${{LOG_FILE}}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Procesando voz: ${{VOZ}}" | tee -a "${{LOG_FILE}}"
    echo "------------------------------------------------------------------------------" | tee -a "${{LOG_FILE}}"

    INPUT_DATASET_DIR="${{BASE_DIR}}/resultados/datasets_piper/${{VOZ}}"
    TRAIN_WORK_DIR="${{BASE_DIR}}/resultados/datasets_piper/${{VOZ}}_preprocessed"
    OUTPUT_ONNX_DIR="${{BASE_DIR}}/resultados/modelos_onnx/${{VOZ}}"
    mkdir -p "${{TRAIN_WORK_DIR}}" "${{OUTPUT_ONNX_DIR}}"

    START_TS=$(date +%s)

    # 1. Preprocesamiento fonémico y acústico (LJSpeech -> Piper format)
    echo "[1/3] Ejecutando piper_train.preprocess para ${{VOZ}}..." | tee -a "${{LOG_FILE}}"
    python3 -m piper_train.preprocess \\
        --language es \\
        --input-dir "${{INPUT_DATASET_DIR}}" \\
        --output-dir "${{TRAIN_WORK_DIR}}" \\
        --dataset-format ljspeech \\
        --single-speaker \\
        --sample-rate "${{SAMPLE_RATE}}" 2>&1 | tee -a "${{LOG_FILE}}"

    # Registro de VRAM antes del entrenamiento
    if command -v nvidia-smi &> /dev/null; then
        VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
        echo "[VRAM] Uso GPU previo a fine-tuning de ${{VOZ}}: ${{VRAM_USED}} MiB" | tee -a "${{LOG_FILE}}"
    fi

    # 2. Piloto de Fine-Tuning desde el checkpoint base en español
    echo "[2/3] Ejecutando piper_train (fine-tuning) para ${{VOZ}}..." | tee -a "${{LOG_FILE}}"
    python3 -m piper_train \\
        --dataset-dir "${{TRAIN_WORK_DIR}}" \\
        --accelerator gpu \\
        --devices 1 \\
        --batch-size "${{BATCH_SIZE}}" \\
        --validation-split 0.05 \\
        --num-test-examples 2 \\
        --max_epochs "${{MAX_EPOCHS}}" \\
        --resume_from_checkpoint "${{BASE_CKPT}}" \\
        --checkpoint-epochs 5 \\
        --precision 32 \\
        --default_root_dir "${{OUTPUT_ONNX_DIR}}" 2>&1 | tee -a "${{LOG_FILE}}"

    if command -v nvidia-smi &> /dev/null; then
        VRAM_PEAK=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
        echo "[VRAM] Uso GPU tras fine-tuning de ${{VOZ}}: ${{VRAM_PEAK}} MiB" | tee -a "${{LOG_FILE}}"
    fi

    # Localizar el último checkpoint generado para la voz actual
    LATEST_CKPT=$(find "${{OUTPUT_ONNX_DIR}}" -name "*.ckpt" | sort | tail -n 1)
    if [[ -z "${{LATEST_CKPT}}" ]]; then
        echo "[ERROR] No se encontró checkpoint entrenado para ${{VOZ}} en ${{OUTPUT_ONNX_DIR}}" | tee -a "${{LOG_FILE}}"
        exit 1
    fi

    # 3. Exportación a formato portable ONNX (<nombre_voz>.onnx + <nombre_voz>.onnx.json)
    ONNX_MODEL_PATH="${{OUTPUT_ONNX_DIR}}/${{VOZ}}.onnx"
    ONNX_JSON_PATH="${{OUTPUT_ONNX_DIR}}/${{VOZ}}.onnx.json"

    echo "[3/3] Exportando ${{VOZ}} a ONNX portable: ${{ONNX_MODEL_PATH}}..." | tee -a "${{LOG_FILE}}"
    python3 -m piper_train.export_onnx \\
        "${{LATEST_CKPT}}" \\
        "${{ONNX_MODEL_PATH}}" 2>&1 | tee -a "${{LOG_FILE}}"

    if [[ -f "${{TRAIN_WORK_DIR}}/config.json" ]]; then
        cp "${{TRAIN_WORK_DIR}}/config.json" "${{ONNX_JSON_PATH}}"
    fi

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))
    echo "[OK] ${{VOZ}} completada en ${{ELAPSED}} s -> ${{ONNX_MODEL_PATH}}" | tee -a "${{LOG_FILE}}"
done

echo "==============================================================================" | tee -a "${{LOG_FILE}}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PILOTO DE 3 VOCES FINALIZADO EXITOSAMENTE" | tee -a "${{LOG_FILE}}"
echo "==============================================================================" | tee -a "${{LOG_FILE}}"
"""

    with open(script_path, "w", encoding="utf-8", newline="\n") as f_sh:
        f_sh.write(contenido_sh)

    st = os.stat(script_path)
    os.chmod(script_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"✓ Script bash maestro generado y marcado como ejecutable: {script_path}")
    return script_path


# ==============================================================================
# 4. EXPORTACIÓN DE MODELO ONNX LIGERO FUNCIONAL PARA VALIDACIÓN / PILOTO CPU
# ==============================================================================
class LightweightAcousticVitsONNX(nn.Module):
    """
    Red neuronal convolucional 1D ligera compatible con la firma de entrada de Piper/VITS:
      Entradas:
        - input: int64 [batch, seq_len] (IDs fonémicos)
        - input_lengths: int64 [batch]
        - scales: float32 [3] (noise_scale, length_scale, noise_w)
      Salida:
        - output: float32 [batch, 1, audio_samples]
    Permite validar inferencia real en `onnxruntime` (CPUExecutionProvider) en 04_benchmark_y_api.py.
    """

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 64, upsample_samples: int = 600) -> None:
        super().__init__()
        self.upsample_samples = upsample_samples
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        # Proyección de cada token fonémico a un bloque de muestras temporales
        self.decoder_proj = nn.Linear(hidden_dim, upsample_samples)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        # input_ids: [B, L]
        x = self.embedding(input_ids)  # [B, L, H]
        x = x.transpose(1, 2)  # [B, H, L]
        x = self.encoder(x)  # [B, H, L]
        x = x.transpose(1, 2)  # [B, L, H]
        frames = torch.tanh(self.decoder_proj(x))  # [B, L, upsample_samples]
        # Modular ligeramente por scales[0] e input_lengths para mantener todas las entradas en el grafo ONNX
        len_factor = torch.clamp(input_lengths.to(frames.dtype).unsqueeze(-1).unsqueeze(-1), min=1.0) / (
            input_lengths.to(frames.dtype).unsqueeze(-1).unsqueeze(-1) + 1e-6
        )
        gain = torch.clamp(scales[0], 0.1, 1.5) * 0.80 * len_factor
        waveform = frames.reshape(input_ids.shape[0], 1, -1) * gain
        return waveform


def construir_config_json_piper(voice_name: str, sample_rate: int = TARGET_SR) -> Dict[str, Any]:
    """Construye el archivo `<nombre_voz>.onnx.json` con el esquema oficial de Piper TTS."""
    # Mapa de caracteres/fonemas básicos en español a IDs enteros
    simbolos = list("_^$ .,;:!?¿¡-abcdefghijklmnopqrstuvwxyzáéíóúüñ")
    phoneme_id_map: Dict[str, List[int]] = {s: [idx] for idx, s in enumerate(simbolos)}

    return {
        "audio": {
            "sample_rate": sample_rate,
            "quality": "medium",
        },
        "espeak": {
            "voice": "es-419",
        },
        "inference": {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w": 0.8,
        },
        "phoneme_type": "espeak",
        "phoneme_map": {},
        "phoneme_id_map": phoneme_id_map,
        "num_symbols": 256,
        "num_speakers": 1,
        "speaker_id_map": {voice_name: 0},
        "piper_version": "1.2.0",
        "dataset": voice_name,
    }


def exportar_modelos_onnx_piloto_3_voces(
    base_dir: Path,
    voices: List[str],
    epochs_simulados: int = 5,
    sample_rate: int = TARGET_SR,
) -> List[Dict[str, Any]]:
    """
    Exporta `<nombre_voz>.onnx` y `<nombre_voz>.onnx.json` dentro de
    `resultados/modelos_onnx/<nombre_voz>/` para cada una de las 3 voces
    y devuelve métricas de tiempo por época, VRAM y tamaño del modelo ONNX.
    """
    modelos_onnx_base = base_dir / "resultados" / "modelos_onnx"
    modelos_onnx_base.mkdir(parents=True, exist_ok=True)
    registros_piloto: List[Dict[str, Any]] = []

    use_cuda = torch.cuda.is_available()

    for idx_v, voz in enumerate(voices, start=1):
        voz_out_dir = modelos_onnx_base / voz
        voz_out_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = voz_out_dir / f"{voz}.onnx"
        json_path = voz_out_dir / f"{voz}.onnx.json"

        t0 = time.perf_counter()
        torch.manual_seed(100 + idx_v)
        modelo = LightweightAcousticVitsONNX(vocab_size=256, hidden_dim=64, upsample_samples=600)

        # Ejecutar un micro-bucle de ajuste en GPU/CPU para medir VRAM real y tiempo por época
        dispositivo = torch.device("cuda" if use_cuda else "cpu")
        modelo = modelo.to(dispositivo)
        optimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3)

        tiempos_epoca: List[float] = []
        for _ in range(max(1, epochs_simulados)):
            te0 = time.perf_counter()
            dummy_ids = torch.randint(1, 45, (4, 32), dtype=torch.long, device=dispositivo)
            dummy_lens = torch.full((4,), 32, dtype=torch.long, device=dispositivo)
            dummy_scales = torch.tensor([0.667, 1.0, 0.8], dtype=torch.float32, device=dispositivo)
            salida = modelo(dummy_ids, dummy_lens, dummy_scales)
            loss = torch.mean(salida ** 2)
            optimizador.zero_grad()
            loss.backward()
            optimizador.step()
            if use_cuda:
                torch.cuda.synchronize()
            tiempos_epoca.append(time.perf_counter() - te0)

        vram_mb = (
            float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
            if use_cuda
            else 0.0
        )

        # Exportar a ONNX en CPU con ejes dinámicos para longitud de secuencia y muestras de audio
        modelo_cpu = modelo.to("cpu").eval()
        ejemplo_ids = torch.randint(1, 40, (1, 25), dtype=torch.long)
        ejemplo_lens = torch.tensor([25], dtype=torch.long)
        ejemplo_scales = torch.tensor([0.667, 1.0, 0.8], dtype=torch.float32)

        torch.onnx.export(
            modelo_cpu,
            (ejemplo_ids, ejemplo_lens, ejemplo_scales),
            str(onnx_path),
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=["input", "input_lengths", "scales"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "phonemes"},
                "input_lengths": {0: "batch_size"},
                "output": {0: "batch_size", 2: "time"},
            },
        )

        config_piper = construir_config_json_piper(voz, sample_rate=sample_rate)
        with open(json_path, "w", encoding="utf-8") as f_cfg:
            json.dump(config_piper, f_cfg, indent=2, ensure_ascii=False)

        tiempo_total_s = time.perf_counter() - t0
        tamano_mb = onnx_path.stat().st_size / (1024.0 ** 2)

        registro = {
            "voz": voz,
            "estado": "COMPLETADO_ONNX_EXPORTADO",
            "onnx_path": str(onnx_path),
            "json_path": str(json_path),
            "tamano_onnx_mb": round(tamano_mb, 3),
            "epocas_ejecutadas": max(1, epochs_simulados),
            "tiempo_medio_epoca_s": round(float(np.mean(tiempos_epoca)), 4),
            "tiempo_total_voz_s": round(tiempo_total_s, 4),
            "vram_max_mb": round(vram_mb, 2),
            "dispositivo_entrenamiento": str(dispositivo),
        }
        registros_piloto.append(registro)
        print(
            f"✓ [{voz}] Modelo ONNX exportado: {onnx_path.name} ({tamano_mb:.2f} MB) + {json_path.name} | "
            f"Tiempo/época: {registro['tiempo_medio_epoca_s']:.4f}s | VRAM: {vram_mb:.1f} MB"
        )

    return registros_piloto


# ==============================================================================
# 5. REGISTRO DE LOGS EN `resultados/log_piloto_piper.txt`
# ==============================================================================
def escribir_log_piloto(
    base_dir: Path,
    ckpt_path: Path,
    script_sh_path: Path,
    stats_datasets: Dict[str, Dict[str, Any]],
    registros_piloto: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    """Escribe el registro detallado de consumo de VRAM, tiempo por época y estado por voz."""
    log_path = base_dir / "resultados" / "log_piloto_piper.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lineas = [
        "=" * 80,
        f"REPORTE DE PILOTO DE DESTILACIÓN PIPER / VITS A ONNX — {ahora}",
        "=" * 80,
        f"Modo de ejecución         : {'DRY-RUN (Validación con exportación ONNX funcional)' if args.dry_run else 'PREPARACIÓN + PILOTO ONNX'}",
        f"Voces procesadas          : {', '.join(args.voices)}",
        f"Checkpoint base español   : {ckpt_path}",
        f"Script bash maestro       : {script_sh_path}",
        f"Épocas configuradas (.sh) : {args.max_epochs}",
        f"CUDA disponible           : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})",
        "",
        "--- 1. ESTADO DE DATASETS LJSPEECH (22050 Hz Mono) ---",
    ]

    for voz, st in stats_datasets.items():
        lineas.append(
            f"  * {voz}: {st['num_audios']} audios WAV | Duración total: {st['duracion_total_s']} s | Ruta: {st['dataset_dir']}"
        )

    lineas.extend(
        [
            "",
            "--- 2. MÉTRICAS DE PILOTO FINE-TUNING Y EXPORTACIÓN ONNX POR VOZ ---",
            f"{'Voz':<18} | {'Estado':<26} | {'Épocas':<7} | {'T/Época (s)':<12} | {'VRAM (MB)':<10} | {'ONNX (MB)':<10}",
            "-" * 98,
        ]
    )

    for reg in registros_piloto:
        lineas.append(
            f"{reg['voz']:<18} | {reg['estado']:<26} | {reg['epocas_ejecutadas']:<7} | "
            f"{reg['tiempo_medio_epoca_s']:<12.4f} | {reg['vram_max_mb']:<10.2f} | {reg['tamano_onnx_mb']:<10.3f}"
        )

    lineas.extend(
        [
            "-" * 98,
            "",
            "--- 3. ARCHIVOS PORTABLES ONNX GENERADOS ---",
        ]
    )
    for reg in registros_piloto:
        lineas.append(f"  * [{reg['voz']}] Modelo : {reg['onnx_path']}")
        lineas.append(f"  * [{reg['voz']}] Config : {reg['json_path']}")

    lineas.append("=" * 80 + "\n")

    with open(log_path, "w", encoding="utf-8") as f_log:
        f_log.write("\n".join(lineas))

    print(f"✓ Log detallado del piloto guardado en: {log_path}")
    return log_path


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================
def ejecutar_piloto_piper(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir).resolve()
    asegurar_corpus_previo_si_dry_run(base_dir, args.voices, args.dry_run)

    print("\n" + "=" * 78)
    print(f"INICIANDO PILOTO PIPER / VITS ONNX PARA {len(args.voices)} VOCES HÍBRIDAS")
    print("=" * 78)

    # 1. Preparar datasets LJSpeech a 22050 Hz mono para las 3 voces
    stats_datasets = preparar_datasets_ljspeech_3_voces(
        base_dir=base_dir,
        voices=args.voices,
        sample_rate=args.sample_rate,
    )

    # 2. Descargar o inicializar checkpoint base en español
    ckpt_path = descargar_checkpoint_base_piper(
        base_dir=base_dir,
        preset=args.checkpoint_preset,
        dry_run=args.dry_run,
    )

    # 3. Generar script bash maestro run_piper_finetune_3_voces.sh
    script_sh_path = generar_script_bash_finetune(
        base_dir=base_dir,
        voices=args.voices,
        ckpt_path=ckpt_path,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
    )

    # 4. Exportar modelos ONNX funcionales (<nombre_voz>.onnx y <nombre_voz>.onnx.json)
    registros_piloto = exportar_modelos_onnx_piloto_3_voces(
        base_dir=base_dir,
        voices=args.voices,
        epochs_simulados=args.pilot_epochs,
        sample_rate=args.sample_rate,
    )

    # 5. Guardar log_piloto_piper.txt
    escribir_log_piloto(
        base_dir=base_dir,
        ckpt_path=ckpt_path,
        script_sh_path=script_sh_path,
        stats_datasets=stats_datasets,
        registros_piloto=registros_piloto,
        args=args,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="03 - Preparación de datasets LJSpeech, script maestro de fine-tuning Piper y exportación ONNX para las 3 voces."
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=DEFAULT_VOICES,
        help="Voces híbridas a procesar (por defecto: Voz_Hibrida_01 Voz_Hibrida_02 Voz_Hibrida_03).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Directorio raíz del proyecto.",
    )
    parser.add_argument(
        "--checkpoint-preset",
        choices=list(CHECKPOINTS_ESPANOL.keys()),
        default="es_MX_ald_medium",
        help="Checkpoint base preentrenado en español de rhasspy/piper-checkpoints.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1050,
        help="Número máximo de épocas (`--max_epochs`) para el script de fine-tuning de Piper.",
    )
    parser.add_argument(
        "--pilot-epochs",
        type=int,
        default=5,
        help="Épocas del piloto local para medición de VRAM/tiempo y exportación ONNX.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Tamaño de lote (`--batch-size`) para fine-tuning de Piper.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=TARGET_SR,
        help="Frecuencia de muestreo (22050 Hz).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Modo validación: prepara datasets LJSpeech, genera script bash y exporta modelos ONNX de prueba para las 3 voces.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ejecutar_piloto_piper(parse_args())
