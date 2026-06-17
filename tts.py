# tts.py — Text-to-speech using Kokoro.
# Replaces the Piper subprocess approach with a direct Python API call.
# The pipeline is loaded lazily on the first call to speak() — no startup cost.

from typing import Optional

import numpy as np
import soundfile as sf

import config

# Pipeline is initialised on first use, not at import time.
_pipeline = None


def _get_pipeline():
    """Return the shared KPipeline, loading it on first call."""
    global _pipeline
    if _pipeline is None:
        print(f"  ⏳  Loading Kokoro voice model ({config.KOKORO_VOICE})...")
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code=config.KOKORO_LANG)
        print("  ✅  Kokoro ready.")
    return _pipeline


def speak(text: str, output_path: str = config.RESPONSE_WAV) -> str:
    """
    Convert text to speech using Kokoro and save to output_path as a WAV file.

    KPipeline returns an iterator of (graphemes, phonemes, audio) tuples.
    Audio chunks are float32 at 24 kHz. They are concatenated and written as
    a 32-bit float WAV so audio.play() can read them without conversion.

    Returns the output path.
    """
    pipeline = _get_pipeline()

    print(f"  \U0001f4dd  TTS input: {len(text)} chars \u2014 \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

    chunks = []
    for i, (_graphemes, _phonemes, audio_chunk) in enumerate(pipeline(
        text,
        voice=config.KOKORO_VOICE,
    )):
        duration = len(audio_chunk) / 24000
        print(f"  \U0001f509  Chunk {i+1}: {len(audio_chunk)} samples / {duration:.2f}s")
        chunks.append(audio_chunk)

    if not chunks:
        # Fallback: 500ms of silence so audio.play() never receives a broken file.
        print("  \u26a0\ufe0f   No chunks generated \u2014 using silence fallback.")
        chunks = [np.zeros(int(0.5 * 24000), dtype=np.float32)]

    audio_data = np.concatenate(chunks)
    total_duration = len(audio_data) / 24000
    print(f"  \U0001f4ca  Total: {len(chunks)} chunk(s) \u00b7 {len(audio_data)} samples \u00b7 {total_duration:.2f}s @ 24000 Hz")

    # Write as 32-bit float — no int16 conversion; audio.play() reads float32.
    sf.write(output_path, audio_data, 24000, subtype="FLOAT")
    return output_path
