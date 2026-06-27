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
import signal

def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt()

signal.signal(signal.SIGTERM, _handle_sigterm)

import audio
import stt
import llm
import tts
import config
import extractor
import memory_manager
import memory
import context_resolver
import fast_path
import logger


def _sanitise(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace — for wake matching."""
    return " ".join(
        "".join(c for c in tok if c.isalnum())
        for tok in text.lower().split()
        if any(c.isalnum() for c in tok)
    )


def voice_engine_loop():
    logger.info("\n🤖  Eleven is ready. Say 'Hey Eleven' to wake her up. Press Ctrl+C to quit.\n")
    logger.debug(f"    Model : {config.OLLAMA_MODEL}")
    logger.debug(f"    STT   : Whisper {config.WHISPER_MODEL}")
    logger.debug(f"    TTS   : ElevenLabs ({config.ELEVENLABS_MODEL_ID})")
    logger.debug(f"    Wake  : STT Polling ({len(config.WAKE_PHRASES)} phrases, {config.WAKE_AWAKE_TIMEOUT_SECS}s session timeout)")
    logger.debug(f"    Listen: {config.VAD_ENGINE.upper()} VAD (stops after {config.VAD_SILENCE_TIMEOUT}s silence)\n")
    logger.debug("─" * 50)

    audio.calibrate_noise()
    memory_manager.load_memory()
    
    import reminders
    reminders.scheduler.start_polling()
    logger.debug("─" * 50)

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
                logger.debug("🔴 Session expired")
                logger.debug("😴 Sleep mode")

            # ── SLEEPING: listen for wake word ────────────────────────────────
            if not session_mode:
                logger.debug("👂 Waiting for wake word")
                wav_in = audio.record_vad()

                wake_result = stt.transcribe(wav_in)
                wake_text = wake_result.text
                if not wake_text:
                    continue

                wake_clean = _sanitise(wake_text)
                logger.debug(f"  🔍  [Wake] Heard: '{wake_clean}'")

                # Match: phrase must be at the START of the utterance.
                matched_phrase = None
                for phrase in config.WAKE_PHRASES:
                    if wake_clean.startswith(phrase):
                        matched_phrase = phrase
                        break

                if not matched_phrase:
                    logger.debug(f"  ✗   [Wake] No match — still sleeping.")
                    continue

                session_mode   = True
                last_interaction = time.monotonic()
                logger.debug(f"  ✅  [Wake] WAKE DETECTED — matched '{matched_phrase}'")
                logger.debug("🟢 Session started")

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
                    logger.debug(f"  📎  Inline prompt: '{user_text}'")
                else:
                    # Prompt comes in the next utterance.
                    logger.debug("  🎙  Listening to prompt...")
                    t0 = time.monotonic()
                    prompt_wav = audio.record_vad()
                    t_rec = time.monotonic() - t0

                    logger.debug("  📝  Transcribing prompt...")
                    t0 = time.monotonic()
                    prompt_result = stt.transcribe(prompt_wav)
                    t_stt = time.monotonic() - t0
                    user_text = prompt_result.text
                    stt_confidence = prompt_result.confidence

                    if not user_text:
                        logger.debug("  (Nothing heard after wake — back to listening)\n")
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
                logger.warning(f"  ⚠️  Suspicious transcript: {reason}")
                logger.debug(f"  🎙  Original: {user_text}")
                logger.debug("  ❓  Confirmation requested")
                response = f"Did you say '{user_text}'?"
                
                t_llm = 0.0
                logger.debug("  🔊  Speaking...")
                t0 = time.monotonic()
                wav_out = tts.speak(response)
                t_tts = time.monotonic() - t0

                t0 = time.monotonic()
                audio.play(wav_out)
                t_play = time.monotonic() - t0

                t_total = time.monotonic() - t_turn
                logger.debug(f"  ⏱   rec={t_rec:.1f}s  stt={t_stt:.1f}s  llm={t_llm:.1f}s  tts={t_tts:.1f}s  play={t_play:.1f}s  total={t_total:.1f}s\n")
                continue
            
            # ── Context Resolution ──────────────────────────────────────────
            t0_ctx = time.monotonic()
            resolved_user_text = context_resolver.resolve_context(user_text, memory.get_history())
            t_context = (time.monotonic() - t0_ctx) * 1000

            
            if resolved_user_text != user_text:
                logger.info(f"  You  : {user_text} -> {resolved_user_text}")
                # Recompute clean_msg for the validator if the text changed
                clean_msg = _sanitise(resolved_user_text)
            else:
                logger.info(f"  You  : {user_text}")
            
            # Use the resolved text for extraction and generation
            user_text = resolved_user_text
            
            # 0. Fast Path — deterministic queries bypass the entire pipeline
            t0_fast = time.monotonic()
            fp_response, fp_tag, fp_sleep = fast_path.route(user_text)
            t_fast_path = (time.monotonic() - t0_fast) * 1000

            if fp_response is not None:
                # Skip intent classification, reminders, tools, LLM, and extractor.
                response = fp_response
                should_sleep = fp_sleep
                t_intent = 0.0
                t_memory_rt = 0.0
                t_reminder = 0.0
                t_tool = 0.0
                t_extract = 0.0
                t_llm = 0.0
                import conversation
                response_time_ms = conversation.calculate_response_time_ms(t_turn, time.monotonic())
                memory.add_turn(user_text, response, response_time_ms=response_time_ms)
                last_interaction = time.monotonic()
                logger.info(f"  Eleven: {response}")
                if should_sleep:
                    logger.debug("  🔊  Speaking goodbye")
                else:
                    logger.debug("  🔊  Speaking...")
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
                    logger.debug("  😴  Sleep mode")
                else:
                    context_manager.update_context_async(user_text, response)
                t_total = time.monotonic() - t_turn
                t_stt_ms = t_stt * 1000
                t_tts_ms = t_tts * 1000
                t_rec_ms = t_rec * 1000
                t_play_ms = t_play * 1000
                t_total_ms = t_total * 1000
                
                logger.info(f"\n[PERF] Total: {t_total:.1f}s\n")
                
                if logger.is_debug():
                    stages = {
                        "Recording": t_rec_ms,
                        "STT": t_stt_ms,
                        "Intent Router": 0.0,
                        "Memory Extraction": 0.0,
                        "Memory Retrieval": 0.0,
                        "Context Build": 0.0,
                        "LLM": 0.0,
                        "TTS Generation": t_tts_ms,
                        "Audio Playback": t_play_ms
                    }
                    slowest = max(stages, key=stages.get)
                    
                    perf_log = ["[PERF]"]
                    for k, v in stages.items():
                        perf_log.append(f"{k}: {v:.0f} ms")
                    perf_log.append(f"Slowest Stage: {slowest}")
                    perf_log.append(f"Total: {t_total_ms:.0f} ms")
                    logger.debug("\n" + "\n".join(perf_log) + "\n")
                    
                continue

            # 1. Intent Router
            t0_intent = time.monotonic()
            import intent_classifier
            import response_policy
            import tool_router
            
            intent = intent_classifier.classify_intent(user_text)
            
            if intent == "GOODBYE":
                logger.debug("  👋  Goodbye detected")
                
            response = None
            should_sleep = False
            t_intent = (time.monotonic() - t0_intent) * 1000
            
            t_memory_rt = 0.0
            t_reminder = 0.0
            t_tool = 0.0
            t_extract = 0.0

            
            if intent in ("GOODBYE", "GREETING", "SMALL_TALK", "ACKNOWLEDGEMENT", "IDENTITY_QUERY", "THANKS"):
                t0_mem_rt = time.monotonic()
                logger.debug(f"[ROUTER] Intent matched: {intent}")
                response, should_sleep = response_policy.apply_policy(intent, user_text)
                t_memory_rt += (time.monotonic() - t0_mem_rt) * 1000
                
            # 2. Reminder Router
            if response is None:
                t0_rem = time.monotonic()
                from reminders import manager as reminder_manager
                rem_intent, rem_response = reminder_manager.route_reminder(user_text)
                if rem_response:
                    logger.debug(f"[ROUTER] Reminder matched: {rem_intent}")
                    response = rem_response
                    t_llm = 0.0
                t_reminder = (time.monotonic() - t0_rem) * 1000
                    
            # 3. Tool Router
            if response is None:
                t0_tool = time.monotonic()
                tool_name, tool_response = tool_router.route_tool(user_text)
                if tool_response:
                    logger.debug(f"[ROUTER] Tool matched: {tool_name}")
                    response = tool_response
                    t_llm = 0.0
                t_tool = (time.monotonic() - t0_tool) * 1000
                    
            # 4. Memory Router
            if response is None and intent.startswith("MEMORY_"):
                t0_mem2 = time.monotonic()
                logger.debug(f"[ROUTER] Memory matched: {intent}")
                response, should_sleep = response_policy.apply_policy(intent, user_text)
                t_memory_rt += (time.monotonic() - t0_mem2) * 1000
            
            # Background Fact Extraction (Applies to Memory intents + Fallback)
            if intent in ("MEMORY_REMEMBER", "MEMORY_UPDATE", "MEMORY_FORGET", "UNKNOWN"):
                t0_ex = time.monotonic()
                extractor.run_extraction_and_save(user_text, stt_confidence)
                t_extract = (time.monotonic() - t0_ex) * 1000
            
            # 4. LLM (Fallback)
            if response is not None:
                t_llm = 0.0
            else:
                    logger.debug("[ROUTER] Falling back to LLM")
                    logger.debug("  📤  [LLM] Sending prompt to LLM...")
                    t0 = time.monotonic()
                    response = llm.chat(user_text)
                    t_llm = time.monotonic() - t0
                    logger.debug("  ✅  [LLM] Response received.")
                    
                    # ── Pre-TTS Response Validator ────────────────────────────
                    # Actively correct the LLM if it hallucinates known core facts.
                    # 1. Name validation
                    if "my name" in clean_msg:
                        stored_name = memory.get_profile().get("name")
                        if stored_name and stored_name.lower() not in response.lower():
                            logger.debug(f"  🛑  [Validator] LLM missed name. Overriding to '{stored_name}'.")
                            response = f"{stored_name}."
                    # 2. Favorite color validation
                    if "favorite color" in clean_msg or "favourite color" in clean_msg:
                        stored_color = memory_manager.get_value("preferences", "favorite_color")
                        if stored_color:
                            import retriever
                            formatted_color = retriever._format_value(stored_color)
                            if formatted_color.lower() not in response.lower():
                                ans = formatted_color[0].upper() + formatted_color[1:] + "."
                                logger.debug(f"  🛑  [Validator] LLM missed color. Overriding to '{ans}'.")
                                response = ans

            # Log memory and TTS start
            import conversation
            response_time_ms = conversation.calculate_response_time_ms(t_turn, time.monotonic())
            memory.add_turn(user_text, response, response_time_ms=response_time_ms)
            last_interaction = time.monotonic()
            if session_mode and not should_sleep:
                logger.debug("⏳ Session timeout reset")
            logger.info(f"  Eleven: {response}")

            if should_sleep:
                logger.debug("  🔊  Speaking goodbye")
            else:
                logger.debug("  🔊  Speaking...")
                
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
                logger.debug("  😴  Sleep mode")
            else:
                context_manager.update_context_async(user_text, response)
            t_total = time.monotonic() - t_turn
            
            # Convert legacy seconds timings to ms for PERF logs
            t_stt_ms = t_stt * 1000
            t_llm_ms = t_llm * 1000 if response is not None else 0.0
            t_tts_ms = t_tts * 1000
            t_rec_ms = t_rec * 1000
            t_play_ms = t_play * 1000
            t_total_ms = t_total * 1000
            
            logger.info(f"\n[PERF] Total: {t_total:.1f}s\n")
            
            if logger.is_debug():
                stages = {
                    "Recording": t_rec_ms,
                    "STT": t_stt_ms,
                    "Intent Router": t_intent,
                    "Memory Extraction": t_extract,
                    "Memory Retrieval": t_memory_rt,
                    "Context Build": t_context,
                    "LLM": t_llm_ms,
                    "TTS Generation": t_tts_ms,
                    "Audio Playback": t_play_ms
                }
                slowest = max(stages, key=stages.get)
                
                perf_log = ["[PERF]"]
                for k, v in stages.items():
                    perf_log.append(f"{k}: {v:.0f} ms")
                perf_log.append(f"Slowest Stage: {slowest}")
                perf_log.append(f"Total: {t_total_ms:.0f} ms")
                logger.debug("\n" + "\n".join(perf_log) + "\n")

        except KeyboardInterrupt:
            logger.info("\n\n👋  Goodbye.\n")
            import context_manager
            import reminders.scheduler
            logger.debug("Stopping background tasks...")
            reminders.scheduler.stop_polling()
            context_manager.stop_async_tasks()
            logger.debug("Exiting cleanly...")
            sys.exit(0)


def main():
    voice_engine_loop()

if __name__ == "__main__":
    main()
