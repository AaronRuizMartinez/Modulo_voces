#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_generar_corpus_prueba.py
===========================
Generador del corpus fonéticamente balanceado (45 frases en español cotidiano y CAA)
para evaluar objetivamente las 3 voces híbridas sintéticas:
  - Voz_Hibrida_01
  - Voz_Hibrida_02
  - Voz_Hibrida_03

Modos soportados:
  1. API Local VoiceStudio compatible con OpenAI (POST /v1/audio/speech).
  2. Inferencia local con Qwen/Qwen3-TTS-12Hz-1.7B-Base + soundfile.
  3. Modo validación (--dry-run) que inicializa carpetas, crea/verifica los
     audios de entrada (3 voces híbridas y 6 donantes) y sintetiza los 135
     audios WAV mono a 22050 Hz con firmas acústicas diferenciadas por hablante.
"""

import argparse
import csv
import io
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import requests
import soundfile as sf
from tqdm import tqdm

# ==============================================================================
# CONFIGURACIÓN DE CARPETAS Y VOCES POR DEFECTO
# ==============================================================================
DEFAULT_VOICES = ["Voz_Hibrida_01", "Voz_Hibrida_02", "Voz_Hibrida_03"]
TARGET_SR = 22050

VOICE_DONOR_MAP: Dict[str, Tuple[str, str]] = {
    "Voz_Hibrida_01": ("Voz_A_01", "Voz_B_01"),
    "Voz_Hibrida_02": ("Voz_A_02", "Voz_B_02"),
    "Voz_Hibrida_03": ("Voz_A_03", "Voz_B_03"),
}

# Parámetros acústicos deterministas para modo --dry-run (f0 y formantes por voz)
SYNTH_PROFILES: Dict[str, Dict[str, float]] = {
    "Voz_A_01": {"f0": 195.0, "f1": 520.0, "f2": 1650.0, "f3": 2650.0, "vibrato": 4.8},
    "Voz_B_01": {"f0": 235.0, "f1": 610.0, "f2": 1920.0, "f3": 2880.0, "vibrato": 5.4},
    "Voz_Hibrida_01": {"f0": 215.0, "f1": 565.0, "f2": 1785.0, "f3": 2765.0, "vibrato": 5.1},
    "Voz_A_02": {"f0": 125.0, "f1": 450.0, "f2": 1420.0, "f3": 2400.0, "vibrato": 4.2},
    "Voz_B_02": {"f0": 155.0, "f1": 495.0, "f2": 1560.0, "f3": 2520.0, "vibrato": 4.6},
    "Voz_Hibrida_02": {"f0": 140.0, "f1": 472.0, "f2": 1490.0, "f3": 2460.0, "vibrato": 4.4},
    "Voz_A_03": {"f0": 210.0, "f1": 580.0, "f2": 1820.0, "f3": 2790.0, "vibrato": 5.2},
    "Voz_B_03": {"f0": 255.0, "f1": 660.0, "f2": 2080.0, "f3": 3020.0, "vibrato": 5.8},
    "Voz_Hibrida_03": {"f0": 232.0, "f1": 620.0, "f2": 1950.0, "f3": 2905.0, "vibrato": 5.5},
}

# ==============================================================================
# 45 FRASES FONÉTICAMENTE BALANCEADAS EN ESPAÑOL COTIDIANO Y CAA
# ==============================================================================
CORPUS_45_FRASES: List[Tuple[str, str]] = [
    # --- Categoría 1: Frases cortas declarativas de CAA (01 - 12) ---
    ("CAA_declarativa", "Hola, buenos días a todos."),
    ("CAA_declarativa", "Quiero tomar un vaso de agua fresca."),
    ("CAA_declarativa", "Necesito ayuda para acomodar mi silla."),
    ("CAA_declarativa", "Me siento muy tranquilo en este lugar."),
    ("CAA_declarativa", "Por favor espera un momento antes de salir."),
    ("CAA_declarativa", "Tengo hambre y quisiera almorzar temprano."),
    ("CAA_declarativa", "Muchas gracias por escucharme con paciencia."),
    ("CAA_declarativa", "Hoy prefiero quedarme descansando en casa."),
    ("CAA_declarativa", "Me duele un poco la cabeza desde la mañana."),
    ("CAA_declarativa", "Quiero llamar a mi familia por teléfono."),
    ("CAA_declarativa", "Ya terminé todas mis actividades del día."),
    ("CAA_declarativa", "Por favor baja el volumen del televisor."),

    # --- Categoría 2: Preguntas frecuentes cotidianas (13 - 23) ---
    ("Pregunta_frecuente", "¿Cómo te encuentras el día de hoy?"),
    ("Pregunta_frecuente", "¿A qué hora vamos a salir para la consulta?"),
    ("Pregunta_frecuente", "¿Puedes repetirme lo que acabas de decir?"),
    ("Pregunta_frecuente", "¿Dónde dejaste las llaves de la puerta principal?"),
    ("Pregunta_frecuente", "¿Qué tenemos preparado para cenar esta noche?"),
    ("Pregunta_frecuente", "¿Me podrías acercar mi tableta de comunicación?"),
    ("Pregunta_frecuente", "¿Cuándo regresan mis hermanos del colegio?"),
    ("Pregunta_frecuente", "¿Hace frío afuera o llevo un abrigo ligero?"),
    ("Pregunta_frecuente", "¿Quién viene a visitarnos este fin de semana?"),
    ("Pregunta_frecuente", "¿Podemos escuchar un poco de música suave?"),
    ("Pregunta_frecuente", "¿Cuánto tiempo falta para llegar al hospital?"),

    # --- Categoría 3: Expresiones emocionales y afectivas (24 - 34) ---
    ("Expresion_emocional", "¡Qué alegría volver a verte después de tanto tiempo!"),
    ("Expresion_emocional", "¡Estoy muy orgulloso de todo lo que logramos juntos!"),
    ("Expresion_emocional", "Me da mucha ilusión participar en este proyecto."),
    ("Expresion_emocional", "¡Qué sorpresa tan bonita me prepararon hoy!"),
    ("Expresion_emocional", "A veces me siento cansado, pero sigo adelante con ánimo."),
    ("Expresion_emocional", "¡Te agradezco de todo corazón por tu apoyo constante!"),
    ("Expresion_emocional", "Me encanta cuando compartimos historias por la tarde."),
    ("Expresion_emocional", "¡Qué excelente noticia acabamos de recibir!"),
    ("Expresion_emocional", "Siento mucha paz cuando escucho el sonido de la lluvia."),
    ("Expresion_emocional", "¡Felicidades por haber alcanzado tu meta con éxito!"),
    ("Expresion_emocional", "Me preocupa llegar tarde, salgamos lo antes posible."),

    # --- Categoría 4: Oraciones largas y fonéticamente ricas (35 - 45) ---
    ("Oracion_larga", "La comunicación aumentativa y alternativa permite expresar pensamientos propios con claridad y autonomía."),
    ("Oracion_larga", "El sistema procesa la señal acústica mediante redes neuronales profundas optimizadas para baja latencia."),
    ("Oracion_larga", "Cada voz sintética conserva una identidad vocal única, natural y agradable durante conversaciones extensas."),
    ("Oracion_larga", "Los dispositivos portátiles funcionan de manera completamente local sin depender de conexiones externas a internet."),
    ("Oracion_larga", "El rápido ferrocarril cruzaba las montañas verdes mientras el sol iluminaba el valle central."),
    ("Oracion_larga", "Para garantizar una excelente inteligibilidad, evaluamos cuidadosamente la pronunciación de cada fonema del español."),
    ("Oracion_larga", "Cuando trabajamos en equipo con dedicación y respeto, superamos cualquier obstáculo técnico con creatividad."),
    ("Oracion_larga", "La tecnología asistiva transforma la vida cotidiana de las personas al brindarles una voz propia y reconocible."),
    ("Oracion_larga", "Durante la sesión de rehabilitación practicamos ejercicios de lectura, respiración y articulación de palabras complejas."),
    ("Oracion_larga", "El modelo ligero en formato portable reduce el consumo de memoria y acelera la respuesta en procesadores estándar."),
    ("Oracion_larga", "Mañana presentaremos los resultados comparativos de estabilidad, similitud acústica y tasa de error de transcripción."),
]


# ==============================================================================
# UTILIDADES DE ESTRUCTURA DE CARPETAS Y AUDIO
# ==============================================================================
def crear_estructura_directorios(base_dir: Path) -> Dict[str, Path]:
    """Crea toda la estructura de directorios requerida por el pipeline."""
    rutas = {
        "donantes": base_dir / "datos_entrada" / "donantes",
        "voces_hibridas": base_dir / "datos_entrada" / "voces_hibridas",
        "modelos_base_piper": base_dir / "modelos_base" / "piper",
        "modelos_onnx": base_dir / "resultados" / "modelos_onnx",
        "corpus_evaluacion": base_dir / "resultados" / "corpus_evaluacion",
        "datasets_piper": base_dir / "resultados" / "datasets_piper",
        "resultados": base_dir / "resultados",
    }
    for ruta in rutas.values():
        ruta.mkdir(parents=True, exist_ok=True)
    return rutas


def normalizar_y_guardar_wav(
    audio: np.ndarray, orig_sr: int, destino: Path, target_sr: int = TARGET_SR
) -> float:
    """Convierte audio a mono float32, remuestrea a target_sr (22050 Hz) y guarda en WAV PCM_16."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = audio.flatten()

    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    pico = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if pico > 0.0:
        audio = (audio / pico) * 0.88

    destino.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destino), audio, target_sr, subtype="PCM_16")
    return float(len(audio) / target_sr)


def importar_audios_existentes_workspace(base_dir: Path) -> None:
    """
    Si existen las carpetas locales Voz1, Voz2, voz3 en el workspace con los audios
    reales del usuario (.wav o .mp3), los importa y convierte automáticamente a WAV
    22050 Hz mono en datos_entrada/donantes y datos_entrada/voces_hibridas.
    """
    mapeo_candidatos = [
        # Voz 1
        (base_dir / "Voz1" / "Voz_A_01.wav", base_dir / "datos_entrada" / "donantes" / "Voz_A_01.wav"),
        (base_dir / "Voz1" / "Voz_B_01.wav", base_dir / "datos_entrada" / "donantes" / "Voz_B_01.wav"),
        (base_dir / "Voz1" / "Voz_Hibrida_01.wav", base_dir / "datos_entrada" / "voces_hibridas" / "Voz_Hibrida_01.wav"),
        # Voz 2
        (base_dir / "Voz2" / "Voz_A_02.wav", base_dir / "datos_entrada" / "donantes" / "Voz_A_02.wav"),
        (base_dir / "Voz2" / "Voz_A_02.mp3", base_dir / "datos_entrada" / "donantes" / "Voz_A_02.wav"),
        (base_dir / "Voz2" / "Voz_B_02.wav", base_dir / "datos_entrada" / "donantes" / "Voz_B_02.wav"),
        (base_dir / "Voz2" / "Voz_Hibrida_02.wav", base_dir / "datos_entrada" / "voces_hibridas" / "Voz_Hibrida_02.wav"),
        # Voz 3
        (base_dir / "voz3" / "Voz_A_03.wav", base_dir / "datos_entrada" / "donantes" / "Voz_A_03.wav"),
        (base_dir / "voz3" / "Voz_B_03.wav", base_dir / "datos_entrada" / "donantes" / "Voz_B_03.wav"),
        (base_dir / "voz3" / "Voz_b_03.mp3", base_dir / "datos_entrada" / "donantes" / "Voz_B_03.wav"),
        (base_dir / "voz3" / "Voz_Hibrida_03.wav", base_dir / "datos_entrada" / "voces_hibridas" / "Voz_Hibrida_03.wav"),
    ]

    for origen, destino in mapeo_candidatos:
        if origen.exists() and not destino.exists():
            try:
                y, sr = librosa.load(str(origen), sr=TARGET_SR, mono=True)
                normalizar_y_guardar_wav(y, sr, destino, TARGET_SR)
                print(f"[INFO] Audio importado y normalizado (22050 Hz mono): {origen.name} -> {destino}")
            except Exception as exc:
                print(f"[WARN] No se pudo importar {origen}: {exc}")


def sintetizar_onda_armonica_dummy(
    speaker_id: str,
    texto: str,
    phrase_idx: int = 0,
    sr: int = TARGET_SR,
    duracion_fija: Optional[float] = None,
) -> np.ndarray:
    """
    Genera una señal armónica pseudo-vocal determinista para modo --dry-run.
    Mantiene la firma espectral consistente del hablante (f0 y formantes) con
    envolvente silábica dependiente del texto para pruebas de embeddings.
    """
    perfil = SYNTH_PROFILES.get(
        speaker_id,
        {"f0": 180.0, "f1": 550.0, "f2": 1700.0, "f3": 2700.0, "vibrato": 5.0},
    )
    n_palabras = max(2, len(texto.split()))
    duracion = duracion_fija if duracion_fija is not None else max(1.6, min(6.5, 0.32 * n_palabras))
    n_samples = int(sr * duracion)
    t = np.linspace(0.0, duracion, n_samples, endpoint=False, dtype=np.float32)

    # Ligera variación prosódica por frase sin perder identidad del locutor
    rng = np.random.default_rng(abs(hash(f"{speaker_id}_{phrase_idx}")) % (2**32))
    f0 = perfil["f0"] * (1.0 + 0.008 * np.sin(2.0 * np.pi * perfil["vibrato"] * t))
    fase_f0 = 2.0 * np.pi * np.cumsum(f0) / sr

    senal = np.zeros_like(t)
    # Armónicos ponderados por cercanía a los 3 formantes del hablante
    for k in range(1, 16):
        freq_k = k * perfil["f0"]
        ganancia_formante = (
            np.exp(-0.5 * ((freq_k - perfil["f1"]) / 180.0) ** 2)
            + 0.75 * np.exp(-0.5 * ((freq_k - perfil["f2"]) / 260.0) ** 2)
            + 0.45 * np.exp(-0.5 * ((freq_k - perfil["f3"]) / 340.0) ** 2)
            + 0.12 / k
        )
        senal += (ganancia_formante / k) * np.sin(k * fase_f0 + (k * 0.17))

    # Modulación de envolvente silábica suave
    tasa_silabica = max(2.5, min(4.8, n_palabras / duracion * 1.6))
    envolvente = 0.65 + 0.35 * np.sin(2.0 * np.pi * tasa_silabica * t + 0.3) ** 2

    # Fade-in y fade-out de 40 ms
    fade_len = min(int(0.04 * sr), n_samples // 4)
    if fade_len > 0:
        envolvente[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        envolvente[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

    ruido_suave = rng.normal(0.0, 0.004, size=n_samples).astype(np.float32)
    audio = (senal * envolvente) + ruido_suave
    pico = float(np.max(np.abs(audio)))
    if pico > 0:
        audio = (audio / pico) * 0.85
    return audio.astype(np.float32)


def asegurar_audios_entrada_dummy(base_dir: Path, voices: List[str], overwrite: bool = False) -> None:
    """
    Garantiza que existan los archivos de las 3 voces híbridas y sus 6 donantes
    en datos_entrada/voces_hibridas y datos_entrada/donantes.
    En --dry-run crea audios dummy si alguno no existe (sin sobrescribir audios reales
    a menos que se indique overwrite=True).
    """
    rutas = crear_estructura_directorios(base_dir)
    importar_audios_existentes_workspace(base_dir)

    for idx_v, voz in enumerate(voices, start=1):
        ruta_hibrida = rutas["voces_hibridas"] / f"{voz}.wav"
        if not ruta_hibrida.exists() or overwrite:
            onda = sintetizar_onda_armonica_dummy(
                speaker_id=voz,
                texto=f"Muestra maestra de referencia vocal para {voz}.",
                phrase_idx=idx_v,
                sr=TARGET_SR,
                duracion_fija=4.0,
            )
            normalizar_y_guardar_wav(onda, TARGET_SR, ruta_hibrida, TARGET_SR)
            print(f"[DRY-RUN] Creado audio maestro dummy: {ruta_hibrida}")

        don_a, don_b = VOICE_DONOR_MAP.get(voz, (f"Voz_A_0{idx_v}", f"Voz_B_0{idx_v}"))
        for donante in (don_a, don_b):
            ruta_donante = rutas["donantes"] / f"{donante}.wav"
            if not ruta_donante.exists() or overwrite:
                onda_d = sintetizar_onda_armonica_dummy(
                    speaker_id=donante,
                    texto=f"Audio de referencia del donante {donante}.",
                    phrase_idx=idx_v * 10,
                    sr=TARGET_SR,
                    duracion_fija=4.0,
                )
                normalizar_y_guardar_wav(onda_d, TARGET_SR, ruta_donante, TARGET_SR)
                print(f"[DRY-RUN] Creado audio donante dummy: {ruta_donante}")


# ==============================================================================
# CLIENTE DE SÍNTESIS (API VOICESTUDIO OPENAI-COMPATIBLE + QWEN3-TTS LOCAL)
# ==============================================================================
class VoiceSynthesisClient:
    """
    Cliente unificado para generación de voz:
      - Prioridad 1: API HTTP compatible con OpenAI de VoiceStudio (POST /v1/audio/speech).
      - Prioridad 2: Inferencia local con Qwen/Qwen3-TTS-12Hz-1.7B-Base.
      - Modo --dry-run: Generación acústica determinista para validación del pipeline.
    """

    def __init__(
        self,
        base_dir: Path,
        api_url: Optional[str] = None,
        mode: str = "auto",
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        dry_run: bool = False,
    ) -> None:
        self.base_dir = base_dir
        self.api_url = api_url or os.environ.get(
            "VOICESTUDIO_API_URL", "http://localhost:8000/v1/audio/speech"
        )
        self.mode = mode
        self.model_name = model_name
        self.dry_run = dry_run
        self.local_model = None
        self._voice_prompts: Dict[str, Any] = {}
        self.api_disponible = False

        if not self.dry_run and self.mode in ("auto", "api"):
            self.api_disponible = self._verificar_api()
            if self.api_disponible:
                print(f"[INFO] Conectado a VoiceStudio API en: {self.api_url}")
            elif self.mode == "api":
                raise RuntimeError(
                    f"No se pudo conectar a VoiceStudio API en {self.api_url} y se forzó --mode api."
                )
            else:
                print(
                    f"[INFO] API VoiceStudio no detectada en {self.api_url}. "
                    "Activando modo alternativo local (Qwen3-TTS)."
                )

    def _verificar_api(self) -> bool:
        """Comprueba si el servidor local de VoiceStudio está escuchando."""
        try:
            parsed_base = self.api_url.split("/v1/")[0]
            resp = requests.get(f"{parsed_base}/docs", timeout=2.0)
            return resp.status_code < 500
        except Exception:
            return False

    def _cargar_modelo_local_qwen(self) -> None:
        """Carga perezosa (lazy loading) de Qwen/Qwen3-TTS-12Hz-1.7B-Base en GPU/CPU."""
        if self.local_model is not None:
            return

        import torch
        from qwen_tts import Qwen3TTSModel

        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dispositivo == "cuda" else torch.float32
        print(f"[INFO] Cargando modelo local {self.model_name} en {dispositivo} ({dtype})...")
        self.local_model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=dispositivo,
            torch_dtype=dtype,
        )

    def sintetizar_a_archivo(
        self,
        texto: str,
        voice_name: str,
        ref_audio_path: Path,
        output_wav_path: Path,
        phrase_idx: int = 0,
        target_sr: int = TARGET_SR,
    ) -> Tuple[float, float, str]:
        """
        Sintetiza `texto` con la voz `voice_name` y guarda el resultado en `output_wav_path`
        como WAV mono a 22050 Hz.
        Retorna: (tiempo_sintesis_s, duracion_audio_s, backend_utilizado)
        """
        t0 = time.perf_counter()

        # 1. Modo --dry-run
        if self.dry_run:
            onda = sintetizar_onda_armonica_dummy(
                speaker_id=voice_name,
                texto=texto,
                phrase_idx=phrase_idx,
                sr=target_sr,
            )
            duracion_s = normalizar_y_guardar_wav(onda, target_sr, output_wav_path, target_sr)
            tiempo_s = time.perf_counter() - t0
            return tiempo_s, duracion_s, "dry_run_dummy"

        # 2. Modo API compatible con OpenAI (VoiceStudio)
        if self.mode == "api" or (self.mode == "auto" and self.api_disponible):
            payload = {
                "model": "qwen3-tts",
                "input": texto,
                "voice": voice_name,
                "response_format": "wav",
                "speed": 1.0,
            }
            try:
                response = requests.post(self.api_url, json=payload, timeout=90)
                response.raise_for_status()
                audio_bytes = io.BytesIO(response.content)
                data, sr = sf.read(audio_bytes, dtype="float32")
                duracion_s = normalizar_y_guardar_wav(data, sr, output_wav_path, target_sr)
                tiempo_s = time.perf_counter() - t0
                return tiempo_s, duracion_s, "voicestudio_openai_api"
            except Exception as api_err:
                if self.mode == "api":
                    raise
                print(f"[WARN] Falló llamada API ({api_err}); conmutando a Qwen3-TTS local...")
                self.api_disponible = False

        # 3. Modo Local con Qwen/Qwen3-TTS-12Hz-1.7B-Base
        import torch

        self._cargar_modelo_local_qwen()
        prompt_voz = self._obtener_prompt_clonacion(voice_name, ref_audio_path)

        outputs = self.local_model.generate_voice_clone(
            text=texto,
            language="Spanish",
            voice_clone_prompt=prompt_voz,
        )

        orig_sr = 24000
        if isinstance(outputs, (tuple, list)):
            audio_data = outputs[0]
            if len(outputs) > 1 and isinstance(outputs[1], (int, float)):
                orig_sr = int(outputs[1])
        else:
            audio_data = outputs

        if isinstance(audio_data, list) and len(audio_data) == 1:
            audio_data = audio_data[0]
        if torch.is_tensor(audio_data):
            audio_data = audio_data.detach().cpu().numpy()
        elif isinstance(audio_data, list):
            audio_data = np.array(audio_data, dtype=np.float32)

        duracion_s = normalizar_y_guardar_wav(audio_data, orig_sr, output_wav_path, target_sr)
        tiempo_s = time.perf_counter() - t0
        return tiempo_s, duracion_s, "qwen3_tts_local"

    def _obtener_prompt_clonacion(self, voice_name: str, ref_audio_path: Path):
        """Extrae y almacena en caché el speaker prompt (x_vector) de cada voz híbrida."""
        if voice_name not in self._voice_prompts:
            if not ref_audio_path.exists():
                raise FileNotFoundError(
                    f"No se encontró el audio de referencia para {voice_name}: {ref_audio_path}"
                )
            self._voice_prompts[voice_name] = self.local_model.create_voice_clone_prompt(
                ref_audio=str(ref_audio_path),
                ref_text="Texto de referencia para inicializar el modelo.",
                x_vector_only_mode=True,
            )
        return self._voice_prompts[voice_name]

    def sintetizar_lote_qwen(
        self,
        textos: List[str],
        voice_name: str,
        ref_audio_path: Path,
        output_wav_paths: List[Path],
        target_sr: int = TARGET_SR,
    ) -> List[Tuple[float, float, str]]:
        """
        Sintetiza un lote de frases en paralelo en GPU usando Qwen3-TTS con el speaker
        embedding precalculado de `voice_name`.
        """
        import torch

        self._cargar_modelo_local_qwen()
        prompt_voz = self._obtener_prompt_clonacion(voice_name, ref_audio_path)

        t0 = time.perf_counter()
        wavs_list, orig_sr = self.local_model.generate_voice_clone(
            text=textos,
            language=["Spanish"] * len(textos),
            voice_clone_prompt=prompt_voz,
        )
        tiempo_lote_s = time.perf_counter() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Distribuir el tiempo de síntesis proporcionalmente a la duración de cada audio del lote
        duraciones: List[float] = []
        for wav_arr, out_path in zip(wavs_list, output_wav_paths):
            if torch.is_tensor(wav_arr):
                wav_arr = wav_arr.detach().cpu().numpy()
            dur_s = normalizar_y_guardar_wav(wav_arr, int(orig_sr), out_path, target_sr)
            duraciones.append(dur_s)

        suma_dur = max(sum(duraciones), 1e-6)
        resultados: List[Tuple[float, float, str]] = []
        for dur_s in duraciones:
            t_indiv = tiempo_lote_s * (dur_s / suma_dur)
            resultados.append((t_indiv, dur_s, "qwen3_tts_local_batch"))
        return resultados


# ==============================================================================
# GENERADOR DE TABLERO HTML INTERACTIVO PARA ESCUCHAR LAS 3 VOCES (45 FRASES)
# ==============================================================================
def generar_tablero_html_corpus(
    base_dir: Path,
    voices: List[str],
    frases_seleccionadas: List[Tuple[str, str]],
    resumen_por_voz: List[Dict[str, object]],
) -> Path:
    """
    Genera `resultados/escuchar_corpus_3_voces.html` con reproductores de audio
    interactivos lado a lado para escuchar las 3 voces híbridas, sus 6 donantes
    y las 45 frases sintetizadas de cada voz.
    """
    html_path = base_dir / "resultados" / "escuchar_corpus_3_voces.html"
    tarjetas_voces_html = []

    for idx_v, voz in enumerate(voices, start=1):
        don_a, don_b = VOICE_DONOR_MAP.get(voz, (f"Voz_A_0{idx_v}", f"Voz_B_0{idx_v}"))
        res_v = next((r for r in resumen_por_voz if r["voz"] == voz), None)
        info_rtf = (
            f"RTF: <strong>{res_v['rtf']:.4f}</strong> | Duración total: <strong>{res_v['duracion_audio_s']:.1f} s</strong>"
            if res_v
            else ""
        )
        tarjetas_voces_html.append(
            f"""
            <div class="voice-card">
                <h3>🎙 {voz}</h3>
                <p class="meta">{info_rtf}</p>
                <div class="audio-block">
                    <span>Muestra Maestra ({voz}.wav):</span>
                    <audio controls preload="none" src="../datos_entrada/voces_hibridas/{voz}.wav"></audio>
                </div>
                <div class="audio-block">
                    <span>Donante A ({don_a}.wav):</span>
                    <audio controls preload="none" src="../datos_entrada/donantes/{don_a}.wav"></audio>
                </div>
                <div class="audio-block">
                    <span>Donante B ({don_b}.wav):</span>
                    <audio controls preload="none" src="../datos_entrada/donantes/{don_b}.wav"></audio>
                </div>
            </div>
            """
        )

    filas_tabla_html = []
    for idx, (categoria, texto) in enumerate(frases_seleccionadas, start=1):
        celdas_audios = []
        for voz in voices:
            archivo = f"{voz}_frase_{idx:03d}.wav"
            rel_wav = f"corpus_evaluacion/{voz}/wavs/{archivo}"
            celdas_audios.append(
                f"""
                <td class="audio-cell">
                    <audio controls preload="none" src="{rel_wav}"></audio>
                    <div class="file-link"><a href="{rel_wav}" download>{archivo}</a></div>
                </td>
                """
            )
        filas_tabla_html.append(
            f"""
            <tr class="phrase-row" data-category="{categoria}">
                <td class="idx-cell">#{idx:03d}</td>
                <td class="cat-cell"><span class="badge badge-{categoria}">{categoria}</span></td>
                <td class="text-cell">{texto}</td>
                {''.join(celdas_audios)}
            </tr>
            """
        )

    headers_voces = "".join(f"<th>{voz}</th>" for voz in voices)

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Comparador Auditivo — 3 Voces Híbridas Sintéticas (45 Frases CAA)</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #38bdf8;
        }}
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        h1, h2, h3 {{ margin-top: 0; }}
        .header {{
            max-width: 1440px;
            margin: 0 auto 24px auto;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px 24px;
        }}
        .cards-grid {{
            max-width: 1440px;
            margin: 0 auto 24px auto;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: 16px;
        }}
        .voice-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 20px;
        }}
        .voice-card h3 {{ color: var(--accent); margin-bottom: 6px; }}
        .voice-card .meta {{ font-size: 0.88rem; color: var(--muted); margin-bottom: 12px; }}
        .audio-block {{ margin-bottom: 10px; }}
        .audio-block span {{ display: block; font-size: 0.82rem; color: var(--muted); margin-bottom: 4px; }}
        audio {{ width: 100%; height: 36px; }}
        .controls-bar {{
            max-width: 1440px;
            margin: 0 auto 16px auto;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            justify-content: space-between;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 18px;
        }}
        .search-input {{
            flex: 1;
            min-width: 260px;
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: #0b1120;
            color: var(--text);
            font-size: 0.95rem;
        }}
        .filter-btn {{
            background: #0b1120;
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 7px 13px;
            cursor: pointer;
            font-size: 0.85rem;
        }}
        .filter-btn.active, .filter-btn:hover {{
            background: var(--accent);
            color: #0f172a;
            font-weight: 600;
        }}
        .table-container {{
            max-width: 1440px;
            margin: 0 auto;
            overflow-x: auto;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            text-align: left;
            vertical-align: middle;
        }}
        th {{
            background: #0b1120;
            color: var(--accent);
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .idx-cell {{ font-weight: 700; color: var(--muted); width: 55px; }}
        .cat-cell {{ width: 155px; }}
        .text-cell {{ min-width: 280px; font-size: 0.96rem; }}
        .audio-cell {{ min-width: 250px; }}
        .file-link {{ font-size: 0.76rem; margin-top: 3px; }}
        .file-link a {{ color: var(--muted); text-decoration: none; }}
        .file-link a:hover {{ color: var(--accent); text-decoration: underline; }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.76rem;
            font-weight: 600;
        }}
        .badge-CAA_declarativa {{ background: #065f46; color: #a7f3d0; }}
        .badge-Pregunta_frecuente {{ background: #1e40af; color: #bfdbfe; }}
        .badge-Expresion_emocional {{ background: #9d174d; color: #fbcfe8; }}
        .badge-Oracion_larga {{ background: #78350f; color: #fde68a; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎧 Comparador Auditivo — 3 Voces Híbridas Sintéticas ({len(frases_seleccionadas)} Frases CAA)</h1>
        <p style="color: var(--muted); margin: 4px 0 0 0;">
            Reproduce y compara lado a lado cada una de las {len(frases_seleccionadas)} frases sintetizadas a 22050 Hz Mono para
            <strong>{', '.join(voices)}</strong>, así como sus muestras maestras y pares de donantes originales.
        </p>
    </div>

    <div class="cards-grid">
        {''.join(tarjetas_voces_html)}
    </div>

    <div class="controls-bar">
        <input type="text" id="searchInput" class="search-input" placeholder="Buscar frase por palabra clave..." oninput="filtrarTabla()" />
        <div>
            <button class="filter-btn active" onclick="seleccionarCategoria('TODAS', this)">Todas ({len(frases_seleccionadas)})</button>
            <button class="filter-btn" onclick="seleccionarCategoria('CAA_declarativa', this)">CAA Declarativas</button>
            <button class="filter-btn" onclick="seleccionarCategoria('Pregunta_frecuente', this)">Preguntas</button>
            <button class="filter-btn" onclick="seleccionarCategoria('Expresion_emocional', this)">Emocionales</button>
            <button class="filter-btn" onclick="seleccionarCategoria('Oracion_larga', this)">Oraciones Largas</button>
        </div>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Categoría</th>
                    <th>Frase en Español</th>
                    {headers_voces}
                </tr>
            </thead>
            <tbody id="corpusTableBody">
                {''.join(filas_tabla_html)}
            </tbody>
        </table>
    </div>

    <script>
        let categoriaActiva = 'TODAS';
        // Pausar otros audios automáticamente al reproducir uno nuevo
        document.addEventListener('play', function(e) {{
            const audios = document.getElementsByTagName('audio');
            for (let i = 0; i < audios.length; i++) {{
                if (audios[i] !== e.target) {{
                    audios[i].pause();
                }}
            }}
        }}, true);

        function seleccionarCategoria(cat, btn) {{
            categoriaActiva = cat;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            filtrarTabla();
        }}

        function filtrarTabla() {{
            const q = document.getElementById('searchInput').value.toLowerCase();
            const filas = document.querySelectorAll('.phrase-row');
            filas.forEach(fila => {{
                const cat = fila.getAttribute('data-category');
                const txt = fila.querySelector('.text-cell').textContent.toLowerCase();
                const coincideCat = (categoriaActiva === 'TODAS' || cat === categoriaActiva);
                const coincideTxt = txt.includes(q);
                fila.style.display = (coincideCat && coincideTxt) ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f_html:
        f_html.write(html_content)
    return html_path


# ==============================================================================
# PIPELINE PRINCIPAL DE GENERACIÓN DE CORPUS
# ==============================================================================
def ejecutar_generacion_corpus(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir).resolve()
    rutas = crear_estructura_directorios(base_dir)
    importar_audios_existentes_workspace(base_dir)

    if args.dry_run:
        asegurar_audios_entrada_dummy(base_dir, args.voices, overwrite=False)

    cliente = VoiceSynthesisClient(
        base_dir=base_dir,
        api_url=args.api_url,
        mode=args.mode,
        model_name=args.model_name,
        dry_run=args.dry_run,
    )

    frases_seleccionadas = CORPUS_45_FRASES[: args.num_phrases]
    registros_tiempos: List[Dict[str, object]] = []
    resumen_por_voz: List[Dict[str, object]] = []

    # Generar vista inicial del comparador HTML desde el inicio
    generar_tablero_html_corpus(
        base_dir=base_dir,
        voices=args.voices,
        frases_seleccionadas=frases_seleccionadas,
        resumen_por_voz=[],
    )

    tiempo_sintesis_global = 0.0
    duracion_audio_global = 0.0
    total_audios_generados = 0

    print("\n" + "=" * 78)
    print(f"INICIANDO GENERACIÓN DE CORPUS ({len(args.voices)} VOCES x {len(frases_seleccionadas)} FRASES)")
    print(f"Modo: {'DRY-RUN (Validación)' if args.dry_run else args.mode} | Sample Rate: {args.sample_rate} Hz Mono")
    print("=" * 78)

    for voz in args.voices:
        voz_dir = rutas["corpus_evaluacion"] / voz
        wavs_dir = voz_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = voz_dir / "metadata.csv"
        ref_audio_path = rutas["voces_hibridas"] / f"{voz}.wav"

        lineas_metadata: List[str] = []
        tiempos_voz: List[float] = []
        duraciones_voz: List[float] = []
        rtfs_voz: List[float] = []
        backend_usado = "desconocido"

        usar_lotes_qwen = (
            not args.dry_run
            and getattr(args, "batch_size", 1) > 1
            and (args.mode == "local" or (args.mode == "auto" and not cliente.api_disponible))
        )

        if usar_lotes_qwen:
            tam_lote = int(getattr(args, "batch_size", 5))
            pbar = tqdm(
                total=len(frases_seleccionadas),
                desc=f"Sintetizando {voz} (GPU batch={tam_lote})",
                unit="audio",
            )
            for inicio in range(0, len(frases_seleccionadas), tam_lote):
                sub_frases = frases_seleccionadas[inicio : inicio + tam_lote]
                textos_lote = [txt for _, txt in sub_frases]
                rutas_lote = [
                    wavs_dir / f"{voz}_frase_{inicio + j + 1:03d}.wav"
                    for j in range(len(sub_frases))
                ]
                res_lote = cliente.sintetizar_lote_qwen(
                    textos=textos_lote,
                    voice_name=voz,
                    ref_audio_path=ref_audio_path,
                    output_wav_paths=rutas_lote,
                    target_sr=args.sample_rate,
                )
                for j, ((categoria, texto), (t_sint, dur_s, backend_usado)) in enumerate(
                    zip(sub_frases, res_lote)
                ):
                    idx = inicio + j + 1
                    nombre_archivo = f"{voz}_frase_{idx:03d}.wav"
                    rtf = t_sint / max(dur_s, 1e-6)
                    tiempos_voz.append(t_sint)
                    duraciones_voz.append(dur_s)
                    rtfs_voz.append(rtf)
                    tiempo_sintesis_global += t_sint
                    duracion_audio_global += dur_s
                    total_audios_generados += 1
                    lineas_metadata.append(f"{nombre_archivo}|{texto}")
                    registros_tiempos.append(
                        {
                            "tipo_registro": "INDIVIDUAL",
                            "voz": voz,
                            "id_frase": f"frase_{idx:03d}",
                            "categoria": categoria,
                            "archivo_wav": nombre_archivo,
                            "num_palabras": len(texto.split()),
                            "num_caracteres": len(texto),
                            "tiempo_sintesis_s": round(t_sint, 6),
                            "duracion_audio_s": round(dur_s, 4),
                            "rtf": round(rtf, 6),
                            "backend": backend_usado,
                        }
                    )
                    pbar.update(1)
                    pbar.set_postfix({"RTF": f"{rtf:.4f}", "dur": f"{dur_s:.2f}s"})
            pbar.close()
        else:
            pbar = tqdm(
                enumerate(frases_seleccionadas, start=1),
                total=len(frases_seleccionadas),
                desc=f"Sintetizando {voz}",
                unit="audio",
            )

            for idx, (categoria, texto) in pbar:
                nombre_archivo = f"{voz}_frase_{idx:03d}.wav"
                ruta_wav = wavs_dir / nombre_archivo

                t_sint, dur_s, backend_usado = cliente.sintetizar_a_archivo(
                    texto=texto,
                    voice_name=voz,
                    ref_audio_path=ref_audio_path,
                    output_wav_path=ruta_wav,
                    phrase_idx=idx,
                    target_sr=args.sample_rate,
                )

                rtf = t_sint / max(dur_s, 1e-6)
                tiempos_voz.append(t_sint)
                duraciones_voz.append(dur_s)
                rtfs_voz.append(rtf)

                tiempo_sintesis_global += t_sint
                duracion_audio_global += dur_s
                total_audios_generados += 1

                # Formato LJSpeech solicitado: nombre_archivo|texto_exacto
                lineas_metadata.append(f"{nombre_archivo}|{texto}")

                registros_tiempos.append(
                    {
                        "tipo_registro": "INDIVIDUAL",
                        "voz": voz,
                        "id_frase": f"frase_{idx:03d}",
                        "categoria": categoria,
                        "archivo_wav": nombre_archivo,
                        "num_palabras": len(texto.split()),
                        "num_caracteres": len(texto),
                        "tiempo_sintesis_s": round(t_sint, 6),
                        "duracion_audio_s": round(dur_s, 4),
                        "rtf": round(rtf, 6),
                        "backend": backend_usado,
                    }
                )
                pbar.set_postfix({"RTF": f"{rtf:.4f}", "dur": f"{dur_s:.2f}s"})

        # Guardar metadata.csv en formato LJSpeech (nombre_archivo|texto_exacto)
        with open(metadata_path, "w", encoding="utf-8", newline="\n") as f_meta:
            for linea in lineas_metadata:
                f_meta.write(linea + "\n")

        suma_t_voz = float(np.sum(tiempos_voz))
        suma_dur_voz = float(np.sum(duraciones_voz))
        rtf_voz_global = suma_t_voz / max(suma_dur_voz, 1e-6)
        rtf_voz_medio = float(np.mean(rtfs_voz))

        resumen_voz = {
            "tipo_registro": "RESUMEN_VOZ",
            "voz": voz,
            "id_frase": "TODAS_45",
            "categoria": "CONSOLIDADO_VOZ",
            "archivo_wav": f"{len(frases_seleccionadas)}_archivos",
            "num_palabras": sum(len(t.split()) for _, t in frases_seleccionadas),
            "num_caracteres": sum(len(t) for _, t in frases_seleccionadas),
            "tiempo_sintesis_s": round(suma_t_voz, 6),
            "duracion_audio_s": round(suma_dur_voz, 4),
            "rtf": round(rtf_voz_global, 6),
            "backend": backend_usado,
        }
        resumen_por_voz.append(resumen_voz)
        registros_tiempos.append(resumen_voz)

        print(
            f"✓ [{voz}] {len(frases_seleccionadas)} audios en {wavs_dir} | "
            f"metadata: {metadata_path.name} | Tiempo total: {suma_t_voz:.2f}s | "
            f"Audio total: {suma_dur_voz:.2f}s | RTF global voz: {rtf_voz_global:.4f} (media: {rtf_voz_medio:.4f})"
        )

    # Registro global consolidado de las 3 voces
    rtf_global = tiempo_sintesis_global / max(duracion_audio_global, 1e-6)
    registro_global = {
        "tipo_registro": "RESUMEN_GLOBAL_3_VOCES",
        "voz": "GLOBAL_3_VOCES",
        "id_frase": f"TOTAL_{total_audios_generados}",
        "categoria": "CONSOLIDADO_GLOBAL",
        "archivo_wav": f"{total_audios_generados}_archivos",
        "num_palabras": sum(r["num_palabras"] for r in resumen_por_voz),
        "num_caracteres": sum(r["num_caracteres"] for r in resumen_por_voz),
        "tiempo_sintesis_s": round(tiempo_sintesis_global, 6),
        "duracion_audio_s": round(duracion_audio_global, 4),
        "rtf": round(rtf_global, 6),
        "backend": resumen_por_voz[0]["backend"] if resumen_por_voz else "N/A",
    }
    registros_tiempos.append(registro_global)

    # Exportar resultados/tiempos_generacion_corpus.csv
    csv_tiempos_path = rutas["resultados"] / "tiempos_generacion_corpus.csv"
    columnas = [
        "tipo_registro",
        "voz",
        "id_frase",
        "categoria",
        "archivo_wav",
        "num_palabras",
        "num_caracteres",
        "tiempo_sintesis_s",
        "duracion_audio_s",
        "rtf",
        "backend",
    ]
    with open(csv_tiempos_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(registros_tiempos)

    html_path = generar_tablero_html_corpus(
        base_dir=base_dir,
        voices=args.voices,
        frases_seleccionadas=frases_seleccionadas,
        resumen_por_voz=resumen_por_voz,
    )

    print("\n" + "=" * 78)
    print(f"RESUMEN GLOBAL DE GENERACIÓN ({total_audios_generados} audios generados):")
    print(f"  - Tiempo total de síntesis : {tiempo_sintesis_global:.4f} s")
    print(f"  - Duración total de audio  : {duracion_audio_global:.4f} s")
    print(f"  - Real-Time Factor (RTF)   : {rtf_global:.6f}")
    print(f"  - Reporte CSV guardado en  : {csv_tiempos_path}")
    print(f"  - Reproductor HTML en      : {html_path}")
    print("=" * 78 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="01 - Generador del corpus de 45 frases para las 3 voces híbridas."
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=DEFAULT_VOICES,
        help="Lista de voces híbridas a procesar (por defecto: Voz_Hibrida_01 Voz_Hibrida_02 Voz_Hibrida_03).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Directorio raíz del proyecto.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("VOICESTUDIO_API_URL", "http://localhost:8000/v1/audio/speech"),
        help="URL del endpoint compatible con OpenAI de VoiceStudio.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "api", "local"],
        default="auto",
        help="Modo de síntesis: 'auto' (intenta API y cae a Qwen3-TTS local), 'api' o 'local'.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Nombre o ruta del modelo Hugging Face para modo local.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=TARGET_SR,
        help="Frecuencia de muestreo de salida en Hz (por defecto: 22050).",
    )
    parser.add_argument(
        "--num-phrases",
        type=int,
        default=45,
        help="Cantidad de frases del corpus a sintetizar por voz (por defecto: 45).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Tamaño de lote para síntesis en GPU con Qwen3-TTS local (por defecto: 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta en modo validación creando audios dummy para las 3 voces, sus 6 donantes y los 135 audios del corpus.",
    )
    return parser.parse_args()


if __name__ == "__main__":
     ejecutar_generacion_corpus(parse_args())
