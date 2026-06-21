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
                import context_manager
                context_manager.clear_context()
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

            # ── Speech Correction Layer ───────────────────────────────────────
            import speech_corrector
            user_text = speech_corrector.correct_transcript(user_text)

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
            
            # ── Tool Calling Layer ──────────────────────────────────────────
            import tool_router
            tool_name, tool_response = tool_router.route_tool(user_text)
            
            response = None
            intent = None
            if tool_response:
                print(f"  🛠  Tool: {tool_name}")
                response = tool_response
                t_llm = 0.0
            else:
                # ── Intent Classification ───────────────────────────────────────
                import intent_classifier
                import response_policy
                
                intent = intent_classifier.classify_intent(user_text)
                print(f"  🧠  Intent: {intent}")
                
                if intent == "GOODBYE":
                    print("  👋  Goodbye detected")
                    
                response, should_sleep = response_policy.apply_policy(intent, user_text)
            
            # Extract facts for new statements (even if we bypass conversational LLM)
            if intent in ("MEMORY_UPDATE", "UNKNOWN"):
                extractor.run_extraction_and_save(user_text, stt_confidence)
            
            if response:
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
            import context_manager
            if should_sleep:
                session_mode = False
                context_manager.clear_context()
                print("  😴  Sleep mode")
            else:
                context_manager.update_context_async(user_text, response)
            t_total = time.monotonic() - t_turn
            print(f"  ⏱   rec={t_rec:.1f}s  stt={t_stt:.1f}s  llm={t_llm:.1f}s  tts={t_tts:.1f}s  play={t_play:.1f}s  total={t_total:.1f}s")
            print()

        except KeyboardInterrupt:
            print("\n\n👋  Goodbye.\n")
            sys.exit(0)


if __name__ == "__main__":
    main()

