# config.py — All tunables in one place.
# Change values here; nothing else needs to be edited.

import os

# Absolute path to the project root (directory containing this file).
# All relative paths are derived from here so the project works
# regardless of the current working directory when python is invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Audio ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000        # Hz — 16kHz is what Whisper expects
RECORD_SECONDS = 5         # How long to listen each turn (used only by legacy record())
CHANNELS = 1               # Mono microphone input
PLAYBACK_BLOCKSIZE = 4096  # samples — ~170ms at 24kHz; prevents CoreAudio underruns

# ── Voice Activity Detection (Silero VAD) ──────────────────────────────────
# Silero VAD is a 2MB neural network that scores each audio chunk with a
# speech probability (0–1). It reliably rejects fan noise, AC hum, keyboard
# clicks, and distant voices — environments where RMS energy detection fails.
#
# VAD_ENGINE: "silero" (recommended) or "energy" (legacy fallback).
VAD_ENGINE            = "silero"

# Silero speech-probability threshold (0.0–1.0).
# Chunks with prob >= this value are classified as speech.
# Lower = more sensitive (picks up quiet speech but more false positives).
# Higher = stricter (fewer false positives but may miss soft speech).
VAD_SILERO_THRESHOLD  = 0.5

# Seconds of continuous silence *after* speech ends before recording stops.
VAD_SILENCE_TIMEOUT   = 1.5

# Ignore speech bursts shorter than this (seconds). Filters out brief noise
# spikes (keyboard tap, click) that pass the probability threshold.
VAD_MIN_SPEECH_DURATION = 0.25

# Minimum total recording duration before silence timeout can fire.
VAD_MIN_DURATION      = 0.5

# Hard cap — prevents infinite recording if VAD never triggers.
VAD_MAX_DURATION      = 30.0

# Audio chunk size fed to Silero. 512 samples @ 16kHz = 32ms — Silero's
# native window size for maximum accuracy. Do not change.
VAD_CHUNK_DURATION    = 0.032   # seconds (512 samples @ 16kHz)

# How long to sample ambient noise at startup for noise floor diagnostics.
# Longer = more accurate floor estimate; shorter = faster startup.
VAD_CALIBRATION_SECONDS = 1.0

# Print per-chunk diagnostics (speech prob, RMS, state). Set False in production.
VAD_DEBUG             = True

# ── Speech-to-Text (Faster-Whisper) ───────────────────────────────────────
# Use multilingual models (no ".en" suffix) for Hindi / Hinglish / English.
# Multilingual quality ranking: medium > small > base > tiny
# ".en" suffix models are English-only and will not recognise Hindi.
WHISPER_MODEL = "small"    # ~466MB; ~3x faster than medium on CPU with acceptable Hindi/Hinglish quality
WHISPER_DEVICE = "cpu"     # CTranslate2 uses Apple Accelerate BLAS on Mac (no Metal)
WHISPER_COMPUTE = "int8"   # int8 = fastest on CPU; reduces model size ~4x

# Language forced to English so Hinglish is always romanized to Latin script.
# This eliminates Malayalam / Norwegian false-detection entirely.
# Whisper in English mode transcribes Hindi words phonetically in Latin script
# (e.g. "haan bilkul", "kya haal hai") — which is natural Hinglish output.
WHISPER_LANGUAGE = "en"

# Vocabulary biasing via initial_prompt.
# Whisper uses this as a fake "previous transcript" to prime the decoder.
# Words that appear here are recognised at higher confidence.
# Edit freely — add any names, tools, or domain terms you use regularly.
WHISPER_INITIAL_PROMPT = (
    "My name is Tushar. I use PHP, Laravel, and Python for development. "
    "I'm talking to Eleven, my AI assistant."
)

# Whisper's internal no-speech gate. If the model's own no-speech probability
# exceeds this value, it returns an empty result rather than hallucinating.
# Works alongside the RMS energy check as a second layer of silence rejection.
WHISPER_NO_SPEECH_THRESHOLD = 0.6

# RMS energy below this value is treated as silence and skipped.
# Range: float32 normalised audio (0.0 – 1.0). 0.01 ≈ –40 dBFS,
# well above typical microphone noise floor (~–60 dBFS).
SILENCE_THRESHOLD = 0.01

# ── LLM (Ollama) ───────────────────────────────────────────────────
OLLAMA_MODEL = "qwen3:4b"  # Optimal: fast enough for voice, good Hinglish quality.
                            # Alternatives: qwen3:1.7b (fastest), qwen3:14b (best quality).

# qwen3 is a reasoning/thinking model. By default it generates a hidden chain-of-thought
# (hundreds of tokens) before the visible response, adding 15–20s of latency per turn.
# Setting OLLAMA_THINK = False disables this. The Python client passes this to Ollama;
# llm.py also injects /no_think into the system prompt as a belt-and-suspenders fallback.
OLLAMA_THINK = False

# Hard cap on tokens generated per response. For a 1–2 sentence conversational reply,
# 120 tokens is generous. This prevents runaway generation if thinking mode is not
# fully suppressed and also keeps latency predictable.
OLLAMA_NUM_PREDICT = 120

# Fallback system prompt used if personality.txt cannot be found.
# The primary personality is defined in personality.txt — edit that file instead.
OLLAMA_SYSTEM_PROMPT = (
    "You are Eleven, a friendly and concise voice AI companion. "
    "Respond in natural Hinglish using Latin script. "
    "Keep responses to 2-3 short sentences. "
    "Do not use Devanagari, bullet points, markdown, or special characters."
)

# ── Text-to-Speech (Kokoro) ─────────────────────────────────────────
KOKORO_VOICE = "hf_alpha"  # Female Hindi voice. Also try: hf_beta
KOKORO_LANG  = "a"         # 'a' = American English phonemizer — correct for Latin-script Hinglish.
                           # 'h' (Hindi) generates garbage phonemes (Q/Y) for English words,
                           # causing Kokoro to output only 0.5s of audio regardless of text length.
KOKORO_SAMPLE_RATE = 24000 # Hz — fixed by the Kokoro model

# ── Temp files ─────────────────────────────────────────────────────────────
# Stored inside the project directory (not /tmp) to avoid macOS Spotlight
# indexing spikes that cause audio underruns at playback start.
_TMP = os.path.join(_HERE, "tmp")
os.makedirs(_TMP, exist_ok=True)

RECORDED_WAV = os.path.join(_TMP, "eleven_input.wav")
RESPONSE_WAV = os.path.join(_TMP, "eleven_output.wav")
