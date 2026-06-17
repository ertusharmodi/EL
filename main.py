# main.py — Eleven Phase 1: local voice AI conversation loop.
# Run: python main.py

import warnings

# ── Suppress harmless third-party startup warnings ───────────────────────────
# These originate inside torch and urllib3; they do not affect functionality.
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer",
    category=UserWarning,
    module=r"torch\.nn\.modules\.rnn",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated",
    category=FutureWarning,
    module=r"torch\.nn\.utils\.weight_norm",
)
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL",
    module=r"urllib3",
)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import time

import audio
import stt
import llm
import tts
import config


def main():
    print("\n🤖  Eleven is ready. Press Ctrl+C to quit.\n")
    print(f"    Model : {config.OLLAMA_MODEL}")
    print(f"    STT   : Whisper {config.WHISPER_MODEL}")
    print(f"    TTS   : Kokoro ({config.KOKORO_VOICE} · lang={config.KOKORO_LANG})")
    print(f"    Listen: {config.VAD_ENGINE.upper()} VAD (stops after {config.VAD_SILENCE_TIMEOUT}s silence)\n")
    print("─" * 50)

    # Measure ambient noise floor once before the first turn.
    # This calibrates the diagnostic output and the energy-fallback threshold.
    audio.calibrate_noise()
    print("─" * 50)
    while True:
        try:
            t_turn = time.monotonic()

            # ── 1. Record ────────────────────────────────────
            t0 = time.monotonic()
            wav_in = audio.record_vad()
            t_rec = time.monotonic() - t0

            # ── 2. Transcribe ────────────────────────────────
            print("  📝  Transcribing...")
            t0 = time.monotonic()
            user_text = stt.transcribe(wav_in)
            t_stt = time.monotonic() - t0

            if not user_text:
                print("  (Nothing heard — try again)\n")
                continue

            print(f"  You  : {user_text}")

            # ── 3. LLM response ──────────────────────────────
            print("  🧠  Thinking...")
            t0 = time.monotonic()
            response = llm.chat(user_text)
            t_llm = time.monotonic() - t0
            print(f"  Eleven: {response}")

            # ── 4. Text-to-speech ────────────────────────────
            print("  🔊  Speaking...")
            t0 = time.monotonic()
            wav_out = tts.speak(response)
            t_tts = time.monotonic() - t0

            # ── 5. Play response ─────────────────────────────
            t0 = time.monotonic()
            audio.play(wav_out)
            t_play = time.monotonic() - t0

            t_total = time.monotonic() - t_turn
            print(f"  ⏱   rec={t_rec:.1f}s  stt={t_stt:.1f}s  llm={t_llm:.1f}s  tts={t_tts:.1f}s  play={t_play:.1f}s  total={t_total:.1f}s")
            print()

        except KeyboardInterrupt:
            print("\n\n👋  Goodbye.\n")
            sys.exit(0)


if __name__ == "__main__":
    main()
