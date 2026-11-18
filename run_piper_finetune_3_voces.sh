#!/usr/bin/env bash
# ==============================================================================
# run_piper_finetune_3_voces.sh
# Script maestro generado automáticamente por 03_piloto_piper_onnx.py
# Ejecuta preprocesamiento, fine-tuning (VITS/Piper) y exportación ONNX
# iterando sobre las 3 voces híbridas: Voz_Hibrida_01 Voz_Hibrida_02 Voz_Hibrida_03
# ==============================================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${BASE_DIR}/resultados/log_piloto_piper.txt"
BASE_CKPT="${BASE_DIR}/modelos_base/piper/es_es_MX_ald_medium/es_MX_ald_medium_base_dummy.ckpt"
MAX_EPOCHS="${MAX_EPOCHS:-1050}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAMPLE_RATE="22050"

mkdir -p "${BASE_DIR}/resultados/modelos_onnx"

echo "==============================================================================" | tee -a "${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INICIO PILOTO FINE-TUNING PIPER ONNX (3 VOCES)" | tee -a "${LOG_FILE}"
echo "Checkpoint base: ${BASE_CKPT} | Max Epochs: ${MAX_EPOCHS} | Batch Size: ${BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"

for VOZ in Voz_Hibrida_01 Voz_Hibrida_02 Voz_Hibrida_03; do
    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Procesando voz: ${VOZ}" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------------------------" | tee -a "${LOG_FILE}"

    INPUT_DATASET_DIR="${BASE_DIR}/resultados/datasets_piper/${VOZ}"
    TRAIN_WORK_DIR="${BASE_DIR}/resultados/datasets_piper/${VOZ}_preprocessed"
    OUTPUT_ONNX_DIR="${BASE_DIR}/resultados/modelos_onnx/${VOZ}"
    mkdir -p "${TRAIN_WORK_DIR}" "${OUTPUT_ONNX_DIR}"

    START_TS=$(date +%s)

    # 1. Preprocesamiento fonémico y acústico (LJSpeech -> Piper format)
    echo "[1/3] Ejecutando piper_train.preprocess para ${VOZ}..." | tee -a "${LOG_FILE}"
    python3 -m piper_train.preprocess \
        --language es \
        --input-dir "${INPUT_DATASET_DIR}" \
        --output-dir "${TRAIN_WORK_DIR}" \
        --dataset-format ljspeech \
        --single-speaker \
        --sample-rate "${SAMPLE_RATE}" 2>&1 | tee -a "${LOG_FILE}"

    # Registro de VRAM antes del entrenamiento
    if command -v nvidia-smi &> /dev/null; then
        VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
        echo "[VRAM] Uso GPU previo a fine-tuning de ${VOZ}: ${VRAM_USED} MiB" | tee -a "${LOG_FILE}"
    fi

    # 2. Piloto de Fine-Tuning desde el checkpoint base en español
    echo "[2/3] Ejecutando piper_train (fine-tuning) para ${VOZ}..." | tee -a "${LOG_FILE}"
    python3 -m piper_train \
        --dataset-dir "${TRAIN_WORK_DIR}" \
        --accelerator gpu \
        --devices 1 \
        --batch-size "${BATCH_SIZE}" \
        --validation-split 0.05 \
        --num-test-examples 2 \
        --max_epochs "${MAX_EPOCHS}" \
        --resume_from_checkpoint "${BASE_CKPT}" \
        --checkpoint-epochs 5 \
        --precision 32 \
        --default_root_dir "${OUTPUT_ONNX_DIR}" 2>&1 | tee -a "${LOG_FILE}"

    if command -v nvidia-smi &> /dev/null; then
        VRAM_PEAK=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
        echo "[VRAM] Uso GPU tras fine-tuning de ${VOZ}: ${VRAM_PEAK} MiB" | tee -a "${LOG_FILE}"
    fi

    # Localizar el último checkpoint generado para la voz actual
    LATEST_CKPT=$(find "${OUTPUT_ONNX_DIR}" -name "*.ckpt" | sort | tail -n 1)
    if [[ -z "${LATEST_CKPT}" ]]; then
        echo "[ERROR] No se encontró checkpoint entrenado para ${VOZ} en ${OUTPUT_ONNX_DIR}" | tee -a "${LOG_FILE}"
        exit 1
    fi

    # 3. Exportación a formato portable ONNX (<nombre_voz>.onnx + <nombre_voz>.onnx.json)
    ONNX_MODEL_PATH="${OUTPUT_ONNX_DIR}/${VOZ}.onnx"
    ONNX_JSON_PATH="${OUTPUT_ONNX_DIR}/${VOZ}.onnx.json"

    echo "[3/3] Exportando ${VOZ} a ONNX portable: ${ONNX_MODEL_PATH}..." | tee -a "${LOG_FILE}"
    python3 -m piper_train.export_onnx \
        "${LATEST_CKPT}" \
        "${ONNX_MODEL_PATH}" 2>&1 | tee -a "${LOG_FILE}"

    if [[ -f "${TRAIN_WORK_DIR}/config.json" ]]; then
        cp "${TRAIN_WORK_DIR}/config.json" "${ONNX_JSON_PATH}"
    fi

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))
    echo "[OK] ${VOZ} completada en ${ELAPSED} s -> ${ONNX_MODEL_PATH}" | tee -a "${LOG_FILE}"
done

echo "==============================================================================" | tee -a "${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PILOTO DE 3 VOCES FINALIZADO EXITOSAMENTE" | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"
