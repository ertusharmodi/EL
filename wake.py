# wake.py — Wake word detection for Eleven using OpenWakeWord.
#
# Responsibilities:
#   - listen_for_wake_word() — block until the configured wake word is detected,
#     then return so the caller can start the conversation turn.
#
# How it works:
#   OpenWakeWord (OWW) runs a small ONNX model on 80ms audio chunks read from the
#   microphone via PyAudio. Each chunk is scored 0.0–1.0 for each loaded wake word
#   model. When the score exceeds WAKE_THRESHOLD the function returns.
#
# Model swap:
#   Set WAKE_MODEL in config.py to a built-in name (e.g. "hey_jarvis") or an
#   absolute path to a .onnx file (e.g. "wake/hey_eleven.onnx"). No code changes
#   are needed — this file reads config.py at import time.
#
# CPU cost: ~1–3% of one core on Apple Silicon at 80ms/chunk cadence.

import os
import sys
import time

import numpy as np
import pyaudio

import config

# ── Lazy model singleton ───────────────────────────────────────────────────────
# OWW model is expensive to load (~0.5s). Load once at first call, reuse forever.
_oww_model = None


def _get_model():
    """Load (or return the cached) OpenWakeWord model."""
    global _oww_model
    if _oww_model is not None:
        return _oww_model

    from openwakeword.model import Model

    model_arg = config.WAKE_MODEL

    # If WAKE_MODEL looks like a file path, load it as a custom ONNX model.
    # Otherwise treat it as a built-in model name.
    if os.path.isfile(model_arg):
        print(f"  👂  [Wake] Loading custom model: {model_arg}")
        _oww_model = Model(wakeword_models=[model_arg], inference_framework="onnx")
    else:
        print(f"  👂  [Wake] Loading built-in model: {model_arg}")
        _oww_model = Model(wakeword_models=[model_arg], inference_framework="onnx")

    return _oww_model


# ── Public API ─────────────────────────────────────────────────────────────────

def listen_for_wake_word() -> None:
    """
    Block until the configured wake word is detected, then return.

    Opens a fresh PyAudio stream each call so it does not conflict with the
    sounddevice stream used by audio.record_vad() and audio.play().
    The stream is always closed before this function returns (normal or error).

    Prints a heartbeat dot every 5 seconds so the operator can confirm the
    process is alive.
    """
    model = _get_model()

    # Derive the key used in OWW's prediction dict from the model setting.
    # Built-in models use their name as the key; custom .onnx files use the
    # basename without extension.
    model_key = config.WAKE_MODEL
    if os.path.isfile(model_key):
        model_key = os.path.splitext(os.path.basename(model_key))[0]

    chunk_samples = int(config.WAKE_SAMPLE_RATE * config.WAKE_CHUNK_MS / 1000)
    chunk_bytes   = chunk_samples * 2   # int16 = 2 bytes per sample

    pa      = pyaudio.PyAudio()
    stream  = None

    try:
        stream = pa.open(
            rate=config.WAKE_SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=chunk_samples,
        )

        print(f"  👂  Listening for wake word ({config.WAKE_MODEL})...", flush=True)

        last_dot  = time.monotonic()
        dot_every = 5.0   # seconds between heartbeat dots

        while True:
            raw = stream.read(chunk_samples, exception_on_overflow=False)

            # OWW expects a 1-D numpy int16 array.
            audio_chunk = np.frombuffer(raw, dtype=np.int16)

            # Run inference. Returns {model_name: score, ...}
            prediction  = model.predict(audio_chunk)

            # Match any key that contains the model name (handles slight name
            # variations between built-in and custom model keys).
            score = 0.0
            for key, val in prediction.items():
                if model_key.lower() in key.lower():
                    score = float(val)
                    break

            if score >= config.WAKE_THRESHOLD:
                print(f"\n  ✅  Wake word detected! (confidence={score:.2f})", flush=True)
                # Reset OWW's internal state so the next listen cycle starts clean.
                model.reset()
                return

            # Heartbeat dot so the terminal shows the process is alive.
            now = time.monotonic()
            if now - last_dot >= dot_every:
                print(".", end="", flush=True)
                last_dot = now

    except KeyboardInterrupt:
        # Propagate so main.py can print the goodbye message.
        raise

    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()
