# stt.py — Speech-to-text using Faster-Whisper.

import warnings
from typing import Optional

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

import config

# Suppress NumPy RuntimeWarnings that leak from Faster-Whisper's mel
# spectrogram computation (divide-by-zero / overflow / invalid in matmul).
# These are internal to the library and harmless when our silence gate is
# in place, but they pollute the terminal output.
warnings.filterwarnings(
    "ignore",
    message=r".*(divide by zero|overflow|invalid value).*matmul.*",
    category=RuntimeWarning,
)

# Model is loaded lazily on the first call to transcribe() rather than at
# import time. This avoids a 2-3 second blocking side effect whenever any
# other module imports stt, and makes the startup sequence more predictable.
_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    """Return the shared WhisperModel, loading it on first call."""
    global _model
    if _model is None:
        import traceback
        print("  ⏳  Starting Whisper initialization...")
        print(f"       model      : {config.WHISPER_MODEL}")
        print(f"       device     : {config.WHISPER_DEVICE}")
        print(f"       compute    : {config.WHISPER_COMPUTE}")
        print(f"       cache dir  : ~/.cache/huggingface/hub")
        print("       Creating WhisperModel instance...")
        try:
            _model = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE,
            )
        except Exception:
            print("\n  ❌  Whisper initialization failed:")
            traceback.print_exc()
            raise
        print("  ✅  Whisper initialization completed.")
    return _model


def _is_silent(wav_path: str) -> bool:
    """
    Return True if the WAV file is too quiet to contain real speech.

    Method: read as float32 (normalised –1.0 to 1.0) and compute RMS energy.
    Anything below SILENCE_THRESHOLD is treated as silence and rejected
    before it ever reaches Whisper, preventing hallucination and the
    divide-by-zero warnings that arise from near-zero mel spectrograms.
    """
    data, _ = sf.read(wav_path, dtype="float32")
    rms = float(np.sqrt(np.mean(data ** 2)))
    return rms < config.SILENCE_THRESHOLD


def transcribe(wav_path: str) -> str:
    """
    Transcribe a WAV file to text.

    Returns the spoken text as a stripped string.
    Returns an empty string if the recording is silent (RMS below threshold)
    or if Whisper's internal no-speech probability exceeds the configured threshold.
    """
    if _is_silent(wav_path):
        return ""

    model = _get_model()
    segments, info = model.transcribe(
        wav_path,
        beam_size=5,
        language=config.WHISPER_LANGUAGE,
        initial_prompt=config.WHISPER_INITIAL_PROMPT,
        condition_on_previous_text=False,       # No history in a single-turn system
        no_speech_threshold=config.WHISPER_NO_SPEECH_THRESHOLD,
    )
    text = " ".join(segment.text for segment in segments).strip()

    # Show transcription confidence. Language is fixed to WHISPER_LANGUAGE
    # so language_probability reflects how confidently Whisper decoded this
    # clip in that language (useful for catching very low-quality recordings).
    print(f"  🎯  Transcribed ({info.language_probability:.0%} confidence)")

    return text
