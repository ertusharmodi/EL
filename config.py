# config.py — All tunables in one place.
# Change values here; nothing else needs to be edited.

import os

from dotenv import load_dotenv

# Load .env from the project root before anything else reads env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_HERE, ".env"))

# Absolute path to the project root — already set above (before dotenv import).

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
#
# IMPORTANT: Include common words you say frequently that Whisper mishears.
# The "age" → "AIDS" error happens because both sound identical to Whisper's
# acoustic model without context. Adding "age", "name", "date" here biases
# the decoder strongly toward the correct word in short ambiguous utterances.
WHISPER_INITIAL_PROMPT = (
    "My name is Tushar. I use PHP, Laravel, Python, and Go for development. "
    "I'm talking to Eleven, my AI companion. "
    "Common words I say: age, name, date, rate, weight, wait, eight, "
    "great, late, state, okay, day, way, play, say, pay, stay. "
    "I speak English and Hinglish."
)

# Whisper's internal no-speech gate. If the model's own no-speech probability
# exceeds this value, it returns an empty result rather than hallucinating.
# Works alongside the RMS energy check as a second layer of silence rejection.
WHISPER_NO_SPEECH_THRESHOLD = 0.6

# Drop segments where the average log-probability is below this threshold.
# -1.0 = drop very low quality segments (likely noise/hallucination).
# -0.5 = stricter; may drop valid low-energy speech.
# Set to None to disable (accept all segments regardless of quality).
WHISPER_LOG_PROB_THRESHOLD = -1.0

# Per-segment confidence threshold for the low-confidence warning flag.
# Segments below this value print a ⚠️ warning — the text is still used
# but the operator knows to treat it with caution.
# Range: 0.0–1.0. 0.6 = flag anything below 60% confidence.
WHISPER_SEGMENT_CONFIDENCE_THRESHOLD = 0.6

# RMS energy below this value is treated as silence and skipped.
# Range: float32 normalised audio (0.0 – 1.0). 0.01 ≈ –40 dBFS,
# well above typical microphone noise floor (~–60 dBFS).
SILENCE_THRESHOLD = 0.01

# ── LLM (Ollama) ───────────────────────────────────────────────────
OLLAMA_MODEL = "qwen3:1.7b"  # Fastest qwen3 model with good Hinglish quality.
                             # qwen3:4b generates 2,200–3,200 thinking tokens per query (~44s).
                             # qwen3:1.7b generates ~500–800 thinking tokens (~10s).
                             # Switch back to qwen3:4b for better quality if latency is acceptable.

# qwen3 is a reasoning/thinking model.  llm.py uses streaming + </think> boundary
# detection to separate reasoning from the final answer in a version-agnostic way.
#
# OLLAMA_NUM_PREDICT is the max total tokens streamed (thinking + answer).
# With streaming, we break as soon as the answer is captured, so this is just
# a safety ceiling.  5000 is enough for any conversational reply even with heavy
# chain-of-thought; set lower (e.g. 2000) if you want a stricter time cap.
OLLAMA_NUM_PREDICT = 5000

# Penalise token repetition. 1.0 = off; 1.3 = strong but safe.
# Prevents qwen3:1.7b from getting stuck in a repetition loop when
# the conversation history contains a repeated response.
OLLAMA_REPEAT_PENALTY = 1.3

# Sampling temperature. 0.8 = natural diversity without hallucination.
# Setting explicitly prevents Ollama's default from varying across versions.
OLLAMA_TEMPERATURE = 0.8


# Fallback system prompt used if personality.txt cannot be found.
# The primary personality is defined in personality.txt — edit that file instead.
OLLAMA_SYSTEM_PROMPT = (
    "You are Eleven, a friendly and concise voice AI companion. "
    "Respond in natural Hinglish using Latin script. "
    "Keep responses to 2-3 short sentences. "
    "Do not use Devanagari, bullet points, markdown, or special characters."
)

# ── Text-to-Speech (ElevenLabs) ────────────────────────────────────────────
# Credentials are read from the .env file at startup (see load_dotenv above).
# They can be overridden at runtime by setting environment variables with the
# same names.
#
# ELEVENLABS_API_KEY  — your secret key from elevenlabs.io/app/settings/api-keys
# ELEVENLABS_VOICE_ID — the voice to use; default is "Rachel" (multilingual).
#                       Browse voices at elevenlabs.io/voice-lab or via the API:
#                       python -c "from elevenlabs.client import ElevenLabs; \
#                                  c=ElevenLabs(); [print(v.voice_id, v.name) for v in c.voices.get_all().voices]"

ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY",  "")   # override via .env
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # "George" — deep, clear, premade (free tier)
# Other free premade voice IDs to try:
#   JBFqnCBsd6RMkjVDRZzb  George    — deep, clear
#   IKne3meq5aSn9XLyUdCD  Charlie   — casual, conversational
#   XB0fDUnXU5powFXDhCwa  Charlotte — warm, female
#   pFZP5JQG7iQjIQuC4Bku  Lily      — bright, female
# Library voices (Rachel, Adam, Josh, etc.) require Starter tier or above.

# Model options (fastest → highest quality):
#   eleven_flash_v2_5   — fastest, lowest latency (~300ms TTFB)
#   eleven_turbo_v2_5   — good balance of speed and quality
#   eleven_multilingual_v2 — highest quality, supports 29 languages
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"

# PCM output format — no lossy codec, no decoding step needed.
# Supported: pcm_16000, pcm_22050, pcm_24000, pcm_44100
# Note: pcm_44100 requires Pro tier or above. pcm_24000 works on all tiers.
ELEVENLABS_OUTPUT_FORMAT = "pcm_24000"

# ── Legacy Text-to-Speech (Kokoro) — attributes kept so main.py banner compiles
KOKORO_VOICE = "hf_alpha"   # no longer used; ElevenLabs is the active TTS
KOKORO_LANG  = "a"          # no longer used
KOKORO_SAMPLE_RATE = 24000  # no longer used

# ── Wake Word (STT Polling) ────────────────────────────────────────────────────
# In STT polling mode, the assistant constantly listens with VAD. When a speech burst
# completes, STT transcribes it. If the transcript matches one of these phrases,
# it wakes up.
WAKE_PHRASES = [
    "hey eleven",
    "hi eleven",
    "hello eleven",
    "hey 11",
    "hi 11",
    "hello 11",
]

# Seconds of inactivity after which the assistant returns to sleep mode.
# The timer resets after every completed LLM response.
WAKE_AWAKE_TIMEOUT_SECS = 30

# ── Memory ─────────────────────────────────────────────────────────────────────
MEMORY_DIR             = os.path.join(_HERE, "memory")
MEMORY_SHORT_TERM_FILE = os.path.join(MEMORY_DIR, "short_term.json")
MEMORY_PROFILE_FILE    = os.path.join(MEMORY_DIR, "profile.json")   # stub for Phase 3
MEMORY_LONG_TERM_FILE  = os.path.join(MEMORY_DIR, "long_term.json")

# Number of conversation turns to keep in short-term memory.
# Each turn = 1 user message + 1 assistant message = 2 JSON entries.
# 20 turns ≈ 400–1200 extra tokens — well within qwen3:1.7b's 32k context.
MEMORY_SHORT_TERM_TURNS = 20

# ── Temp files ─────────────────────────────────────────────────────────────
# Stored inside the project directory (not /tmp) to avoid macOS Spotlight
# indexing spikes that cause audio underruns at playback start.
_TMP = os.path.join(_HERE, "tmp")
os.makedirs(_TMP, exist_ok=True)

RECORDED_WAV = os.path.join(_TMP, "eleven_input.wav")
RESPONSE_WAV = os.path.join(_TMP, "eleven_output.wav")
