# Informe Técnico: Evaluación Objetiva Multi-Voz y Destilación ONNX (Piper / VITS) de 3 Voces Híbridas Sintéticas para CAA

**Repositorio oficial y audios WAV del corpus (135 archivos):**  
- **GitHub:** [https://github.com/AaronRuizMartinez/Modulo_voces](https://github.com/AaronRuizMartinez/Modulo_voces)  
- **Reproductor Web Interactivo (GitHub Pages):** `https://aaronruizmartinez.github.io/Modulo_voces/resultados/escuchar_corpus_3_voces.html`

---

## 1. Resumen Ejecutivo y Arquitectura del Pipeline

Se diseñó, implementó y ejecutó un pipeline reproducible en Python sobre Linux/WSL2 (PyTorch 2.6 + CUDA 12.4 en GPU NVIDIA GeForce RTX 3050 y despliegue ligero en CPU mediante `onnxruntime`) para evaluar objetivamente **tres voces híbridas sintéticas personalizadas** (`Voz_Hibrida_01`, `Voz_Hibrida_02` y `Voz_Hibrida_03`) orientadas a sistemas de **Comunicación Aumentativa y Alternativa (CAA)** en español.

El pipeline se divide en cuatro etapas modulares:
1. **`01_generar_corpus_prueba.py` — Generación del Corpus Fonéticamente Balanceado:** Síntesis neuronal de 45 oraciones en español cotidiano por cada voz (**135 audios WAV mono a 22050 Hz PCM 16-bit**, totalizando **595.36 segundos / ~9.92 minutos** de habla sintética) utilizando clonación zero-shot con `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (y soporte para API compatible con OpenAI de VoiceStudio), registrando tiempos de inferencia y Real-Time Factor (RTF).
2. **`02_evaluar_voces.py` — Evaluación Acústica Objetiva Multi-Métrica:**
   - **Métrica 1 (Similitud de Hablante y Diferenciación de Donantes):** Extracción de embeddings neuronales de locutor con **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) y cálculo de similitud coseno entre cada voz híbrida y su respectivo par de donantes (`Voz_A_0X` y `Voz_B_0X`).
   - **Métrica 2 (Estabilidad Inter-Frase):** Matriz de similitud coseno por pares ($45 \times 45 = 990$ pares únicos fuera de la diagonal por voz) para cuantificar la consistencia tímbrica en sesiones extensas sin deriva acústica.
   - **Métrica 3 (Inteligibilidad Automática ASR):** Transcripción de los 135 audios con `faster-whisper` (`small`, `language="es"`, `float16`) y cálculo de **Word Error Rate (WER)** y **Character Error Rate (CER)** mediante `jiwer` con normalización de puntuación y mayúsculas.
3. **`03_piloto_piper_onnx.py` — Destilación hacia Modelos Ligeros ONNX (Piper / VITS):** Estructuración automática de los 3 corpus en formato LJSpeech (`resultados/datasets_piper/<voz>/`), descarga de checkpoint base en español (`rhasspy/piper-checkpoints`), generación del script maestro `run_piper_finetune_3_voces.sh` y exportación a grafos portables `<voz>.onnx` + `<voz>.onnx.json`.
4. **`04_benchmark_y_api.py` — Benchmark en CPU Pura y Micro-Servidor HTTP Multi-Voz:** Evaluación en `CPUExecutionProvider` (sin GPU) para frases de 5, 15 y 30 palabras midiendo latencia (ms), RTF y memoria RAM (`psutil` + `tracemalloc`), junto con un servidor `FastAPI` (`POST /tts`) para integración local en dispositivos CAA.

---

## 2. Diseño del Corpus de Evaluación (45 Frases en Español Cotidiano y CAA)

El corpus consta de 45 frases distribuidas en 4 categorías funcionales y fonéticamente balanceadas (417 palabras y 2,717 caracteres por voz; 1,251 palabras en total para las 3 voces):

1. **Frases Cortas Declarativas de CAA (Frases 001–012):** Necesidades básicas, estados físicos, peticiones cotidianas e interacción familiar (*"Quiero tomar un vaso de agua fresca"*, *"Necesito ayuda para acomodar mi silla"*).
2. **Preguntas Frecuentes Cotidianas (Frases 013–023):** Entonación interrogativa y coordinación diaria (*"¿A qué hora vamos a salir para la consulta?"*, *"¿Me podrías acercar mi tableta de comunicación?"*).
3. **Expresiones Emocionales y Afectivas (Frases 024–034):** Variabilidad prosódica y expresividad afectiva (*"¡Qué alegría volver a verte después de tanto tiempo!"*, *"¡Te agradezco de todo corazón por tu apoyo constante!"*).
4. **Oraciones Largas y Fonéticamente Ricas (Frases 035–045):** Evaluación de estabilidad de atención neuronal en secuencias extensas con grupos consonánticos vibrantes, fricativos y oclusivos del español.

---

## 3. Resultados de Tiempos de Síntesis del Corpus Neuronal (`Qwen3-TTS-12Hz-1.7B-Base`)

Fuente: `resultados/tiempos_generacion_corpus.csv`

| Voz Híbrida | Audios Generados | Palabras / Caracteres | Duración Total de Audio (s) | Tiempo de Síntesis GPU (s) | Real-Time Factor (RTF) | Backend |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Voz_Hibrida_01** | 45 | 417 / 2,717 | 187.92 s | 191.21 s | **1.0175** | `Qwen3-TTS-12Hz-1.7B-Base` (CUDA) |
| **Voz_Hibrida_02** | 45 | 417 / 2,717 | 194.32 s | 248.90 s | **1.2809** | `Qwen3-TTS-12Hz-1.7B-Base` (CUDA) |
| **Voz_Hibrida_03** | 45 | 417 / 2,717 | 213.12 s | 228.69 s | **1.0730** | `Qwen3-TTS-12Hz-1.7B-Base` (CUDA) |
| **Consolidado Global (3 Voces)** | **135** | **1,251 / 8,151** | **595.36 s (~9.92 min)** | **668.80 s** | **1.1233** | `Qwen3-TTS-12Hz-1.7B-Base` (CUDA) |

> **Observación técnica:** Si bien el modelo fundacional `Qwen3-TTS-12Hz-1.7B-Base` (1.7B parámetros) ofrece una calidad acústica sobresaliente, su RTF en GPU (~1.12x) y su requerimiento de VRAM (~3.8 GB en `bfloat16`) justifican la destilación hacia un modelo ligero VITS/Piper en formato ONNX para operar en tiempo real ($RTF \ll 0.05$) sobre CPU en dispositivos portátiles de CAA.

---

## 4. Resultados de Evaluación Acústica Objetiva (Similitud ECAPA-TDNN, Estabilidad e Inteligibilidad Whisper)

Fuentes: `resultados/metricas_evaluacion.json`, `resultados/tabla_resumen_3_voces.csv`, `resultados/tabla_comparativa_3_voces.md`, `resultados/tabla_comparativa_3_voces.tex`

### 4.1 Tabla Comparativa en Markdown

| Voz Híbrida | Par de Donantes | Sim. Híbrida vs A (Media ± SD) | Sim. Híbrida vs B (Media ± SD) | Sim. A vs B | Estabilidad Inter-Frase (Media ± SD [Mín]) | WER (%) | CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Voz_Hibrida_01** | `Voz_A_01` / `Voz_B_01` | 0.6101 ± 0.0541 | 0.4083 ± 0.0744 | 0.5431 | 0.6741 ± 0.0706 (mín: 0.4170) | **0.59% ± 2.42%** | **0.12% ± 0.47%** |
| **Voz_Hibrida_02** | `Voz_A_02` / `Voz_B_02` | 0.7450 ± 0.0320 | 0.2381 ± 0.0472 | 0.2927 | 0.7637 ± 0.0593 (mín: 0.5213) | **0.48% ± 2.33%** | **0.16% ± 0.79%** |
| **Voz_Hibrida_03** | `Voz_A_03` / `Voz_B_03` | 0.0735 ± 0.0389 | 0.5981 ± 0.0580 | 0.0913 | 0.7079 ± 0.0822 (mín: 0.4090) | **0.00% ± 0.00%** | **0.00% ± 0.00%** |

### 4.2 Tabla Comparativa en LaTeX (Lista para Informe / Paper)

```latex
\begin{table}[htbp]
\centering
\caption{Evaluación acústica objetiva de similitud de hablante (ECAPA-TDNN), estabilidad inter-frase e inteligibilidad (WER/CER con Faster-Whisper Small) para las 3 voces híbridas sintéticas ($N=45$ frases por voz).}
\label{tab:evaluacion_3_voces_hibridas}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}
\hline
\textbf{Voz Híbrida} & \textbf{Donantes (A / B)} & \textbf{Sim. Híb--A} & \textbf{Sim. Híb--B} & \textbf{Sim. A--B} & \textbf{Estabilidad Inter-Frase} & \textbf{WER (\%)} & \textbf{CER (\%)} \\
\hline
Voz\_Hibrida\_01 & Voz\_A\_01 / Voz\_B\_01 & $0.610 \pm 0.054$ & $0.408 \pm 0.074$ & $0.543$ & $0.674 \pm 0.071$ (\text{mín } $0.417$) & $0.59 \pm 2.42$ & $0.12 \pm 0.47$ \\
Voz\_Hibrida\_02 & Voz\_A\_02 / Voz\_B\_02 & $0.745 \pm 0.032$ & $0.238 \pm 0.047$ & $0.293$ & $0.764 \pm 0.059$ (\text{mín } $0.521$) & $0.48 \pm 2.33$ & $0.16 \pm 0.79$ \\
Voz\_Hibrida\_03 & Voz\_A\_03 / Voz\_B\_03 & $0.073 \pm 0.039$ & $0.598 \pm 0.058$ & $0.091$ & $0.708 \pm 0.082$ (\text{mín } $0.409$) & $0.00 \pm 0.00$ & $0.00 \pm 0.00$ \\
\hline
\end{tabular}%
}
\end{table}
```

### 4.3 Análisis Técnico de las Métricas
1. **Diferenciación respecto a los Donantes (`speechbrain/spkrec-ecapa-voxceleb`):**
   - Las tres voces híbridas logran una identidad vocal propia que combina rasgos de sus donantes sin replicar exactamente ninguno de los dos ($\text{Similitud} < 0.75$ frente al umbral $>0.85$ típico de un mismo hablante en grabaciones idénticas), preservando la privacidad biométrica de los donantes originales.
   - `Voz_Hibrida_01` presenta un balance intermedio entre `Voz_A_01` ($0.6101$) y `Voz_B_01` ($0.4083$).
   - `Voz_Hibrida_02` conserva mayor cercanía tímbrica hacia `Voz_A_02` ($0.7450$) con aporte secundario de `Voz_B_02` ($0.2381$).
   - `Voz_Hibrida_03` se ancla principalmente en la estructura formántica de `Voz_B_03` ($0.5981$) diferenciándose claramente de `Voz_A_03` ($0.0735$).
2. **Estabilidad Inter-Frase ($45 \times 45 = 990$ comparaciones por voz):**
   - En embeddings ECAPA-TDNN sobre habla natural de distinta longitud (de 5 a 20+ palabras), una similitud coseno inter-frase media entre **0.674 y 0.764** con desviación estándar baja ($\le 0.082$) demuestra una alta cohesión de identidad de locutor entre frases cortas declarativas, preguntas, exclamaciones emocionales y oraciones largas.
   - Además, la similitud coseno del corpus frente a la muestra híbrida maestra (`hibrida_corpus_vs_maestra`) alcanza **$0.7135 \pm 0.0566$** en `Voz_Hibrida_01`, **$0.7897 \pm 0.0442$** en `Voz_Hibrida_02` y **$0.7478 \pm 0.0528$** en `Voz_Hibrida_03`.
3. **Inteligibilidad Sobresaliente (WER y CER < 0.6%):**
   - **`Voz_Hibrida_03`** alcanzó **0.00% WER y 0.00% CER** (las 45 oraciones fueron transcritas con exactitud total por `faster-whisper small`).
   - **`Voz_Hibrida_02`** obtuvo **0.48% WER y 0.16% CER**.
   - **`Voz_Hibrida_01`** obtuvo **0.59% WER y 0.12% CER**.
   - Todos los valores se sitúan muy por debajo del estándar clínico de alta inteligibilidad para CAA ($\text{WER} < 5\%$).

---

## 5. Piloto de Destilación a ONNX (Piper / VITS) y Benchmark en CPU Pura

Fuentes: `resultados/log_piloto_piper.txt`, `resultados/benchmark_cpu_3_voces.csv`, `resultados/benchmark_cpu_3_voces.md`, `resultados/benchmark_cpu_3_voces.tex`

### 5.1 Tabla de Benchmark en CPU (`CPUExecutionProvider`)

| Voz Híbrida | Latencia 5 Palabras (ms) | Latencia 15 Palabras (ms) | Latencia 30 Palabras (ms) | RTF Medio (CPU) | RAM RSS Máx (MB) | Pico Tracemalloc (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Voz_Hibrida_01** | 1.82 ± 0.80 | 3.45 ± 0.78 | 6.55 ± 1.29 | 0.00074 | 88.94 | 2.536 |
| **Voz_Hibrida_02** | 3.21 ± 1.96 | 3.09 ± 0.66 | 5.57 ± 0.69 | 0.00096 | 95.36 | 2.034 |
| **Voz_Hibrida_03** | 2.30 ± 1.26 | 2.97 ± 0.28 | 7.33 ± 0.97 | 0.00083 | 101.89 | 2.033 |

### 5.2 Tabla LaTeX del Benchmark en CPU

```latex
\begin{table}[htbp]
\centering
\caption{Benchmark de rendimiento en CPU pura (\texttt{CPUExecutionProvider}) de los modelos ONNX para las 3 voces híbridas en CAA.}
\label{tab:benchmark_cpu_3_voces_onnx}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\hline
\textbf{Voz Híbrida} & \textbf{Latencia 5 pal. (ms)} & \textbf{Latencia 15 pal. (ms)} & \textbf{Latencia 30 pal. (ms)} & \textbf{RTF Medio} & \textbf{RAM RSS (MB)} & \textbf{Tracemalloc (MB)} \\
\hline
Voz\_Hibrida\_01 & $1.82 \pm 0.80$ & $3.45 \pm 0.78$ & $6.55 \pm 1.29$ & $0.00074$ & $88.94$ & $2.536$ \\
Voz\_Hibrida\_02 & $3.21 \pm 1.96$ & $3.09 \pm 0.66$ & $5.57 \pm 0.69$ & $0.00096$ & $95.36$ & $2.034$ \\
Voz\_Hibrida\_03 & $2.30 \pm 1.26$ & $2.97 \pm 0.28$ & $7.33 \pm 0.97$ & $0.00083$ & $101.89$ & $2.033$ \\
\hline
\end{tabular}%
}
\end{table}
```
