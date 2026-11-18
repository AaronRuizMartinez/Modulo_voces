# Pipeline de Evaluación Objetiva Multi-Voz (3 Voces Híbridas) y Destilación ONNX (Piper / VITS) para CAA

Este repositorio implementa el flujo de trabajo completo en Python (Linux / WSL2 con PyTorch + CUDA y despliegue ONNX 100% en CPU) para evaluar objetivamente las **3 voces híbridas sintéticas** (`Voz_Hibrida_01`, `Voz_Hibrida_02` y `Voz_Hibrida_03`), cruzarlas contra sus respectivos pares de donantes (`Voz_A_01`/`Voz_B_01`, `Voz_A_02`/`Voz_B_02`, `Voz_A_03`/`Voz_B_03`), destilarlas a modelos ligeros `.onnx` compatibles con **Piper TTS** y servirlas mediante una API HTTP local para dispositivos de **Comunicación Aumentativa y Alternativa (CAA)** sin dependencia de GPU en producción.

---

## 1. Estructura de Carpetas y Archivos

```text
Modulo_voces/
├── requirements.txt
├── 01_generar_corpus_prueba.py
├── 02_evaluar_voces.py
├── 03_piloto_piper_onnx.py
├── 04_benchmark_y_api.py
├── run_piper_finetune_3_voces.sh        # Generado por 03_piloto_piper_onnx.py
├── README_ejecucion.md
├── datos_entrada/
│   ├── donantes/
│   │   ├── Voz_A_01.wav
│   │   ├── Voz_B_01.wav
│   │   ├── Voz_A_02.wav
│   │   ├── Voz_B_02.wav
│   │   ├── Voz_A_03.wav
│   │   └── Voz_B_03.wav
│   └── voces_hibridas/
│       ├── Voz_Hibrida_01.wav
│       ├── Voz_Hibrida_02.wav
│       └── Voz_Hibrida_03.wav
├── modelos_base/
│   └── piper/                           # Checkpoint preentrenado en español (rhasspy/piper-checkpoints)
└── resultados/
    ├── corpus_evaluacion/               # 45 WAVs (22050 Hz mono) + metadata.csv por cada una de las 3 voces
    │   ├── Voz_Hibrida_01/wavs/
    │   ├── Voz_Hibrida_02/wavs/
    │   └── Voz_Hibrida_03/wavs/
    ├── datasets_piper/                  # Datasets formateados en LJSpeech listos para piper_train
    │   ├── Voz_Hibrida_01/
    │   ├── Voz_Hibrida_02/
    │   └── Voz_Hibrida_03/
    ├── modelos_onnx/                    # Modelos portables <voz>.onnx y <voz>.onnx.json
    │   ├── Voz_Hibrida_01/
    │   ├── Voz_Hibrida_02/
    │   └── Voz_Hibrida_03/
    ├── tiempos_generacion_corpus.csv
    ├── metricas_evaluacion.json
    ├── tabla_resumen_3_voces.csv
    ├── tabla_comparativa_3_voces.md
    ├── tabla_comparativa_3_voces.tex
    ├── log_piloto_piper.txt
    ├── benchmark_cpu_3_voces.csv
    ├── benchmark_cpu_3_voces.md
    ├── benchmark_cpu_3_voces.tex
    └── benchmark_cpu_3_voces.json
```

---

## 2. Instalación de Dependencias

Se recomienda utilizar el entorno virtual con PyTorch + CUDA (`/home/aaron/Qwen3-TTS/.venv` o uno nuevo):

```bash
# Opción A: Usar el entorno existente con PyTorch + CUDA y Qwen3-TTS
source /home/aaron/Qwen3-TTS/.venv/bin/activate
pip install -r requirements.txt

# Opción B: Crear un entorno virtual dedicado
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Validación Rápida del Pipeline Completo (`--dry-run`)

Todos los scripts soportan `--dry-run` y `--voices Voz_Hibrida_01 Voz_Hibrida_02 Voz_Hibrida_03` para verificar la estructura de carpetas, importar o crear audios dummy para las 3 voces y sus 6 donantes, sintetizar las 135 frases de prueba y exportar todas las tablas CSV, Markdown, JSON y LaTeX:

```bash
python3 01_generar_corpus_prueba.py --dry-run
python3 02_evaluar_voces.py --dry-run
python3 03_piloto_piper_onnx.py --dry-run
python3 04_benchmark_y_api.py --dry-run --benchmark
```

---

## 4. Ejecución en Producción (GPU + Audio Real)

### Paso 1: Generación del Corpus Balanceado (45 frases x 3 voces = 135 audios)
- Si VoiceStudio está activo (`http://localhost:8000/v1/audio/speech`), el script lo consume automáticamente.
- Si desea configurar otro puerto/URL: `export VOICESTUDIO_API_URL="http://localhost:8000/v1/audio/speech"`.
- Si la API no está levantada, conmuta automáticamente al modo local usando `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.

```bash
# Modo automático (intenta API VoiceStudio -> fallback local Qwen3-TTS)
python3 01_generar_corpus_prueba.py --mode auto

# Forzar inferencia local directa con Qwen/Qwen3-TTS-12Hz-1.7B-Base en GPU
python3 01_generar_corpus_prueba.py --mode local
```

### Paso 2: Evaluación Acústica Objetiva de las 3 Voces
Calcula **Similitud de Hablante** (`speechbrain/spkrec-ecapa-voxceleb` o `resemblyzer`), **Estabilidad Inter-Frase** (matriz de similitud coseno $45 \times 45$) e **Inteligibilidad** (`faster-whisper` + `jiwer` WER/CER):

```bash
python3 02_evaluar_voces.py --embedding-backend auto --whisper-model small
```

### Paso 3: Preparación de Datasets LJSpeech y Piloto Piper / VITS a ONNX
Prepara `resultados/datasets_piper/<nombre_voz>/`, descarga el checkpoint base en español desde Hugging Face (`rhasspy/piper-checkpoints`), genera `run_piper_finetune_3_voces.sh` y exporta los modelos `.onnx`:

```bash
python3 03_piloto_piper_onnx.py --checkpoint-preset es_MX_ald_medium --max-epochs 1050

# Para ejecutar el fine-tuning completo de Piper en bucle sobre las 3 voces:
./run_piper_finetune_3_voces.sh
```

### Paso 4: Benchmark en CPU Pura y Micro-Servidor HTTP FastAPI (`POST /tts`)

```bash
# Ejecutar benchmark en CPUExecutionProvider (5, 15 y 30 palabras + RTF + RAM)
python3 04_benchmark_y_api.py --benchmark --repeticiones 10

# Levantar el servidor HTTP local multi-voz en el puerto 8080 (100% CPU)
python3 04_benchmark_y_api.py --serve --host 0.0.0.0 --port 8080
```

Ejemplo de consumo del endpoint `POST /tts` con `curl`:

```bash
curl -X POST "http://localhost:8080/tts" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hola, quiero tomar un vaso de agua fresca.", "voice": "Voz_Hibrida_01"}' \
     --output salida_voz_01.wav
```
