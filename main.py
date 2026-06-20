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
import extractor
import memory_manager
import memory
import context_resolver


def _sanitise(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace — for wake matching."""
    return " ".join(
        "".join(c for c in tok if c.isalnum())
        for tok in text.lower().split()
        if any(c.isalnum() for c in tok)
    )


def main():
    print("\n🤖  Eleven is ready. Say 'Hey Eleven' to wake her up. Press Ctrl+C to quit.\n")
    print(f"    Model : {config.OLLAMA_MODEL}")
    print(f"    STT   : Whisper {config.WHISPER_MODEL}")
    print(f"    TTS   : ElevenLabs ({config.ELEVENLABS_MODEL_ID})")
    print(f"    Wake  : STT Polling ({len(config.WAKE_PHRASES)} phrases, {config.WAKE_AWAKE_TIMEOUT_SECS}s session timeout)")
    print(f"    Listen: {config.VAD_ENGINE.upper()} VAD (stops after {config.VAD_SILENCE_TIMEOUT}s silence)\n")
    print("─" * 50)

    audio.calibrate_noise()
    memory_manager.load_memory()
    print("─" * 50)

    # ── Session state ─────────────────────────────────────────────────────────
    session_mode   = False
    last_interaction = 0.0    # monotonic timestamp of last LLM response
    SESSION_TIMEOUT = 60.0

    while True:
        try:
            now = time.monotonic()

            # ── Check session timeout ─────────────────────────────────────────
            if session_mode and (now - last_interaction) >= SESSION_TIMEOUT:
                session_mode = False
                print("🔴 Session expired")
                print("😴 Sleep mode")

            # ── SLEEPING: listen for wake word ────────────────────────────────
            if not session_mode:
                print("👂 Waiting for wake word")
                wav_in = audio.record_vad()

                wake_result = stt.transcribe(wav_in)
                wake_text = wake_result.text
                if not wake_text:
                    continue

                wake_clean = _sanitise(wake_text)
                print(f"  🔍  [Wake] Heard: '{wake_clean}'")

                # Match: phrase must be at the START of the utterance.
                matched_phrase = None
                for phrase in config.WAKE_PHRASES:
                    if wake_clean.startswith(phrase):
                        matched_phrase = phrase
                        break

                if not matched_phrase:
                    print(f"  ✗   [Wake] No match — still sleeping.")
                    continue

                session_mode   = True
                last_interaction = time.monotonic()
                print(f"  ✅  [Wake] WAKE DETECTED — matched '{matched_phrase}'")
                print("🟢 Session started")

                # Inline prompt: user spoke prompt in same breath as wake phrase.
                inline_text = wake_clean[len(matched_phrase):].strip()
                for prefix in ("and ", ", "):
                    if inline_text.startswith(prefix):
                        inline_text = inline_text[len(prefix):].strip()

                if inline_text:
                    user_text = inline_text
                    stt_confidence = wake_result.confidence
                    t_rec = 0.0
                    t_stt = 0.0
                    print(f"  📎  Inline prompt: '{user_text}'")
                else:
                    # Prompt comes in the next utterance.
                    print("  🎙  Listening to prompt...")
                    t0 = time.monotonic()
                    prompt_wav = audio.record_vad()
                    t_rec = time.monotonic() - t0

                    print("  📝  Transcribing prompt...")
                    t0 = time.monotonic()
                    prompt_result = stt.transcribe(prompt_wav)
                    t_stt = time.monotonic() - t0
                    user_text = prompt_result.text
                    stt_confidence = prompt_result.confidence

                    if not user_text:
                        print("  (Nothing heard after wake — back to listening)\n")
                        continue

            # ── AWAKE: any speech goes directly to LLM ────────────────────────
            else:
                t0 = time.monotonic()
                wav_in = audio.record_vad()
                t_rec = time.monotonic() - t0

                t0 = time.monotonic()
                prompt_result = stt.transcribe(wav_in)
                t_stt = time.monotonic() - t0

                user_text = prompt_result.text
                stt_confidence = prompt_result.confidence

                if not user_text:
                    continue

            # Remove old EXIT_COMMANDS check to let Intent Classifier handle it
            clean_msg = _sanitise(user_text)
            
            should_sleep = False
            
            # ── Common: prompt → LLM → TTS → play ────────────────────────────
            t_turn = time.monotonic()
            
            # ── Speech Sanity Check ─────────────────────────────────────────
            import speech_sanity
            is_suspicious, reason = speech_sanity.check_sanity(user_text, t_rec, stt_confidence)
            if is_suspicious:
                print(f"  ⚠️  Suspicious transcript: {reason}")
                print(f"  🎙  Original: {user_text}")
                print("  ❓  Confirmation requested")
                response = f"Did you say '{user_text}'?"
                
                t_llm = 0.0
                print("  🔊  Speaking...")
                t0 = time.monotonic()
                wav_out = tts.speak(response)
                t_tts = time.monotonic() - t0

                t0 = time.monotonic()
                audio.play(wav_out)
                t_play = time.monotonic() - t0

                t_total = time.monotonic() - t_turn
                print(f"  ⏱   rec={t_rec:.1f}s  stt={t_stt:.1f}s  llm={t_llm:.1f}s  tts={t_tts:.1f}s  play={t_play:.1f}s  total={t_total:.1f}s\n")
                continue
            
            # ── Context Resolution ──────────────────────────────────────────
            resolved_user_text = context_resolver.resolve_context(user_text, memory.get_history())
            
            if resolved_user_text != user_text:
                print(f"  You  : {user_text} -> {resolved_user_text}")
                # Recompute clean_msg for the validator if the text changed
                clean_msg = _sanitise(resolved_user_text)
            else:
                print(f"  You  : {user_text}")
            
            # Use the resolved text for extraction and generation
            user_text = resolved_user_text
            
            # ── Intent Classification ───────────────────────────────────────
            import intent_classifier
            intent = intent_classifier.classify_intent(user_text)
            print(f"  🧠  Intent: {intent}")
            
            response = None
            if intent == "GREETING":
                if "good morning" in clean_msg:
                    response = "Good morning."
                else:
                    response = "Hi."
            elif intent == "THANKS":
                response = "You're welcome."
            elif intent == "GOODBYE":
                print("  👋  Goodbye detected")
                response = "See you later."
                should_sleep = True
            elif intent == "IDENTITY":
                response = "I'm Eleven."
            elif intent == "SMALL_TALK":
                if "how are you" in clean_msg or "how's it going" in clean_msg:
                    response = "I'm good. How about you?"
                else:
                    response = "Just talking with you."
            elif intent == "ACKNOWLEDGEMENT":
                if "i'll do it" in clean_msg or "ill do it" in clean_msg:
                    response = "Sounds good."
                else:
                    response = "Alright."
            
            if response:
                t_llm = 0.0
            else:
                # Extract facts before generating the response
                extractor.run_extraction_and_save(user_text, stt_confidence)
                
                import retriever
                direct_ans = retriever.retrieve_direct_answer(user_text)
                
                if direct_ans:
                    response = direct_ans
                    t_llm = 0.0
                else:
                    print("  📤  [LLM] Sending prompt to LLM...")
                    t0 = time.monotonic()
                    response = llm.chat(user_text)
                    t_llm = time.monotonic() - t0
                    print("  ✅  [LLM] Response received.")
                    
                    # ── Pre-TTS Response Validator ────────────────────────────
                    # Actively correct the LLM if it hallucinates known core facts.
                    # 1. Name validation
                    if "my name" in clean_msg:
                        stored_name = memory.get_profile().get("name")
                        if stored_name and stored_name.lower() not in response.lower():
                            print(f"  🛑  [Validator] LLM missed name. Overriding to '{stored_name}'.")
                            response = f"{stored_name}."
                    # 2. Favorite color validation
                    if "favorite color" in clean_msg or "favourite color" in clean_msg:
                        stored_color = memory_manager.get_value("preferences", "favorite_color")
                        if stored_color:
                            import retriever
                            formatted_color = retriever._format_value(stored_color)
                            if formatted_color.lower() not in response.lower():
                                ans = formatted_color[0].upper() + formatted_color[1:] + "."
                                print(f"  🛑  [Validator] LLM missed color. Overriding to '{ans}'.")
                                response = ans

            # Log memory and TTS start
            memory.add_turn(user_text, response)
            last_interaction = time.monotonic()
            if session_mode and not should_sleep:
                print("⏳ Session timeout reset")
            print(f"  Eleven: {response}")

            if should_sleep:
                print("  🔊  Speaking goodbye")
            else:
                print("  🔊  Speaking...")
                
            t0 = time.monotonic()
            wav_out = tts.speak(response)
            t_tts = time.monotonic() - t0

            t0 = time.monotonic()
            audio.play(wav_out)
            t_play = time.monotonic() - t0

            if should_sleep:
                session_mode = False
                print("  😴  Sleep mode")

            t_total = time.monotonic() - t_turn
            print(f"  ⏱   rec={t_rec:.1f}s  stt={t_stt:.1f}s  llm={t_llm:.1f}s  tts={t_tts:.1f}s  play={t_play:.1f}s  total={t_total:.1f}s")
            print()

        except KeyboardInterrupt:
            print("\n\n👋  Goodbye.\n")
            sys.exit(0)


if __name__ == "__main__":
    main()

