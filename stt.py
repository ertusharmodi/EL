# stt.py — Speech-to-text using Faster-Whisper.
#
# Key design choices:
#   - Language forced to "en" so Hinglish is always romanised to Latin script.
#   - initial_prompt biases the vocabulary toward known words/names.
#   - condition_on_previous_text=False prevents hallucination drift across segments.
#   - Per-segment confidence is logged so low-quality transcriptions are visible.
#   - A low-confidence warning is printed when any segment falls below the threshold.

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

    Confidence logging:
        Per-segment avg_logprob and no_speech_prob are printed for every segment.
        A ⚠️ warning is printed when a segment's confidence falls below
        WHISPER_SEGMENT_CONFIDENCE_THRESHOLD so questionable transcriptions
        are visible in the terminal before they reach the LLM.
    """
    if _is_silent(wav_path):
        return ""

    model = _get_model()
    segments_gen, info = model.transcribe(
        wav_path,
        beam_size=5,
        language=config.WHISPER_LANGUAGE,
        initial_prompt=config.WHISPER_INITIAL_PROMPT,
        condition_on_previous_text=False,   # prevents hallucination drift across segments
        no_speech_threshold=config.WHISPER_NO_SPEECH_THRESHOLD,
        log_prob_threshold=config.WHISPER_LOG_PROB_THRESHOLD,  # drop very low-prob segments
        vad_filter=True,                    # built-in VAD removes silent padding before decoding
        vad_parameters=dict(
            min_silence_duration_ms=300,    # ms of silence to split segments
        ),
    )

    # Materialise the generator so we can inspect each segment.
    segments = list(segments_gen)

    parts = []
    low_confidence_flags = []

    for i, seg in enumerate(segments):
        seg_text = seg.text.strip()
        if not seg_text:
            continue

        # avg_logprob is log-probability of the segment tokens (higher = more confident).
        # Convert to a 0–1 confidence score for display: exp(avg_logprob).
        seg_conf = float(min(1.0, max(0.0, 2 ** seg.avg_logprob)))  # log2 → probability
        is_low = seg_conf < config.WHISPER_SEGMENT_CONFIDENCE_THRESHOLD

        print(
            f"  🎙  Segment {i+1}: conf={seg_conf:.0%}  "
            f"no_speech={seg.no_speech_prob:.2f}  "
            f"{'⚠️  LOW CONFIDENCE' if is_low else ''}  "
            f"\"{seg_text}\""
        )

        if is_low:
            low_confidence_flags.append(seg_text)

        parts.append(seg_text)

    text = " ".join(parts).strip()

    # Summary line: overall language confidence + low-confidence warning
    lang_conf = info.language_probability
    if low_confidence_flags:
        print(
            f"  ⚠️   Low-confidence segment(s) detected — "
            f"transcription may contain errors. "
            f"Flagged: {low_confidence_flags}"
        )
        print(f"  🎯  Transcribed ({lang_conf:.0%} language confidence) [REVIEW ADVISED]")
    else:
        print(f"  🎯  Transcribed ({lang_conf:.0%} language confidence)")

    return text
