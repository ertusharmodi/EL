# tts.py — Text-to-speech using ElevenLabs.
# Replaces the Kokoro pipeline with a call to the ElevenLabs REST API
# via the official Python SDK.  The public function signature is unchanged:
#
#   speak(text: str, output_path: str = config.RESPONSE_WAV) -> str
#
# ElevenLabs returns raw PCM audio (no WAV header).  We wrap it with a
# standard WAV header using the `wave` module so that soundfile / sounddevice
# in audio.py can read it without any modification.

import os
import struct
import wave
from typing import Optional

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

import config
import logger

# ── Load .env once at import time ──────────────────────────────────────────
# dotenv does not overwrite variables that are already in the environment,
# so running with a real env var set will always take precedence.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── ElevenLabs client singleton ────────────────────────────────────────────
# Initialised lazily on first call to speak() — no cost at import time.
_client: Optional[ElevenLabs] = None


def _get_client() -> ElevenLabs:
    """Return the shared ElevenLabs client, creating it on first call."""
    global _client
    if _client is None:
        api_key = os.getenv("ELEVENLABS_API_KEY") or config.ELEVENLABS_API_KEY
        if not api_key:
            raise RuntimeError(
                "ElevenLabs API key not found.  "
                "Set ELEVENLABS_API_KEY in .env or as an environment variable."
            )
        _client = ElevenLabs(api_key=api_key)
        logger.debug("  ✅  ElevenLabs client ready.")
    return _client


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, num_channels: int = 1, sample_width: int = 2) -> bytes:
    """
    Wrap raw signed-16-bit PCM bytes in a RIFF/WAV container.

    Parameters
    ----------
    pcm_bytes    : raw PCM audio (little-endian, signed 16-bit)
    sample_rate  : e.g. 44100, 24000, 16000
    num_channels : 1 = mono, 2 = stereo
    sample_width : bytes per sample (2 for int16)

    Returns
    -------
    Complete WAV file as bytes, ready to write to disk.
    """
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def speak(text: str, output_path: str = config.RESPONSE_WAV) -> str:
    """
    Convert *text* to speech using ElevenLabs and save to *output_path* as WAV.

    The ElevenLabs API returns raw PCM audio (pcm_44100 format by default).
    We collect all chunks, wrap them in a WAV header, and write the result so
    that audio.play() — which uses soundfile — can read the file unchanged.

    Returns the output path.
    """
    client = _get_client()

    voice_id  = os.getenv("ELEVENLABS_VOICE_ID") or config.ELEVENLABS_VOICE_ID
    model_id  = config.ELEVENLABS_MODEL_ID
    fmt       = config.ELEVENLABS_OUTPUT_FORMAT   # e.g. "pcm_44100"
    # Derive sample rate from the format string ("pcm_44100" → 44100)
    sample_rate = int(fmt.split("_")[1])

    logger.debug(
        f"  📝  TTS input: {len(text)} chars — "
        f"\"{text[:80]}{'...' if len(text) > 80 else ''}\""
    )
    logger.debug(f"  🔊  ElevenLabs → voice={voice_id}  model={model_id}  fmt={fmt}")

    # Stream audio from ElevenLabs
    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=fmt,
    )

    # Collect PCM chunks
    chunks: list[bytes] = []
    total_bytes = 0
    for chunk in audio_stream:
        if chunk:
            chunks.append(chunk)
            total_bytes += len(chunk)

    if not chunks:
        # Fallback: 500ms of silence (int16 zeros) so audio.play() never
        # receives a broken file.
        logger.warning("  ⚠️   No audio received — using silence fallback.")
        silence_samples = int(0.5 * sample_rate)
        chunks = [b"\x00\x00" * silence_samples]
        total_bytes = len(chunks[0])

    pcm_bytes = b"".join(chunks)
    duration  = total_bytes / (sample_rate * 2)   # 2 bytes per int16 sample
    logger.debug(
        f"  📊  Total: {len(chunks)} chunk(s) · {total_bytes:,} bytes "
        f"· {duration:.2f}s @ {sample_rate} Hz"
    )

    # Wrap in WAV container and write
    wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate=sample_rate)
    with open(output_path, "wb") as fh:
        fh.write(wav_bytes)

    logger.debug(f"  ✅  Saved WAV → {output_path}")
    return output_path
