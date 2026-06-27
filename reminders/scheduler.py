import threading
import time
from datetime import datetime
import dateutil.parser
from reminders import storage
import tts
import audio
import logger
import memory

COMMON_VERBS = {
    "drink", "call", "push", "go", "eat", "buy", "pay", "send", "write",
    "read", "clean", "wash", "text", "email", "finish", "start", "renew",
    "check", "review", "get", "make", "take", "do", "wake", "sleep", "feed",
    "walk", "run", "workout", "exercise", "study", "work", "meet", "pick",
    "drop", "cancel", "book", "order", "cook", "prepare", "fix", "update"
}

IMPORTANT_KEYWORDS = {"mom", "dad", "boss", "doctor", "flight", "urgent", "meeting", "deadline"}


def generate_reminder_speech(task: str) -> str:
    words = task.strip().split()
    if not words:
        return "Just a reminder."

    first_word = words[0].lower()

    if first_word in COMMON_VERBS:
        speech = f"Time to {task}."
    else:
        speech = f"Just a reminder: {task}."

    # Check importance
    task_lower = task.lower()
    if any(k in task_lower for k in IMPORTANT_KEYWORDS):
        name = memory.get_profile().get("name")
        if name:
            speech = f"{name}, {speech[0].lower()}{speech[1:]}"

    return speech


def start_polling():
    """Starts a daemon thread to poll reminders."""
    thread = threading.Thread(target=_poll_loop, daemon=True)
    thread.start()


def _poll_loop():
    while True:
        try:
            logger.debug("[REMINDER] Checking reminders...")
            active_reminders = storage.list_reminders()
            now = datetime.now()

            for r in active_reminders:
                due_dt = dateutil.parser.parse(r["due_at"])
                if now >= due_dt:
                    logger.debug("[REMINDER] Due reminder found")
                    logger.info(f"[REMINDER] Triggered: {r['title']}")

                    # Speak
                    text_to_speak = generate_reminder_speech(r['title'])
                    try:
                        wav_out = tts.speak(text_to_speak)
                        audio.play(wav_out)
                    except Exception as e:
                        logger.warning(f"  ⚠️  [Reminders TTS Error]: {e}")

                    # Save to history
                    storage.add_to_history(r, "completed")

                    # Mark it complete
                    storage.complete_reminder(partial_title=r["title"])
                    logger.debug("[REMINDER] Marked completed")

        except Exception as e:
            logger.warning(f"  ⚠️  [Reminders Scheduler Error]: {e}")

        time.sleep(5)
