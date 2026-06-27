import logger
import re
import dateparser
from typing import Tuple, Optional
from reminders import storage

def parse_time_and_task(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts time string and task string from "remind me <time> to <task>".
    """
    clean = text.lower().strip()
    if not clean.startswith("remind me"):
        return None, None
        
    # Remove "remind me "
    clean = clean[len("remind me "):].strip()
    
    # Split on " to "
    # We want to find the LAST " to " to allow tasks that contain "to", 
    # but normally the time comes first: "in 2 hours to push my code"
    # Wait, "remind me to push my code in 2 hours" is also common.
    # Let's handle both.
    
    # Try: "remind me [time] to [task]"
    # It's safer to split on the first " to " if it's early in the string, or we can just try parsing until we hit " to ".
    # A simple regex: ^(.+?) to (.+)$
    
    match = re.match(r"^(.+?)\s+to\s+(.+)$", clean, re.IGNORECASE)
    if match:
        time_str = match.group(1).strip()
        task_str = match.group(2).strip()
        
        # Validate that dateparser can understand the time_str
        dt = dateparser.parse(time_str, settings={'PREFER_DATES_FROM': 'future'})
        if dt:
            return time_str, task_str
            
        # If it failed, maybe the user said "remind me to [task] [time]"
        # E.g. "remind me to go to the gym tomorrow at 9 AM"
        # Since this is harder to split without an NLP model, we will just stick to 
        # the required format: "Remind me in 2 hours to push my code"
        
    return None, None

def route_reminder(user_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Evaluates if the user_text is a reminder command.
    Returns (intent_name, response_string) if handled, otherwise (None, None).
    """
    clean = user_text.lower().strip().strip(".!?")
    
    # 1. CREATE
    if clean.startswith("remind me"):
        time_str, task_str = parse_time_and_task(user_text)
        if time_str and task_str:
            dt = dateparser.parse(time_str, settings={'PREFER_DATES_FROM': 'future'})
            if dt:
                storage.create_reminder(task_str, dt.isoformat())
                logger.debug(f"  ⏰  [Reminders] Created: '{task_str}' due at {dt.isoformat()}")
                return "REMINDER_CREATE", "Reminder created."
        # If parsing fails, we might just want to let it fall through to LLM or return error.
        # But the requirement says "No LLM calls for reminder commands."
        return "REMINDER_CREATE", "I couldn't understand the time for the reminder."

    # 2. LIST
    if clean in ("what are my reminders", "show reminders", "list reminders"):
        active = storage.list_reminders()
        count = len(active)
        logger.debug(f"  ⏰  [Reminders] List requested. Found {count}.")
        if count == 0:
            return "REMINDER_LIST", "You have no reminders."
        elif count == 1:
            return "REMINDER_LIST", "You have 1 reminder."
        else:
            return "REMINDER_LIST", f"You have {count} reminders."
            
    # Helper to extract index or title
    def _extract_target(text: str, prefix: str) -> Tuple[Optional[int], Optional[str]]:
        content = text[len(prefix):].strip()
        
        # If the whole text contained "reminder 1", try to find a digit
        match = re.search(r"\b(\d+)\b", content)
        if match and "reminder" in text:
            # We assume if they say "reminder 1", they mean index
            return int(match.group(1)), None
            
        # e.g. "my gym reminder" -> "gym"
        content = content.replace("my ", "").replace("reminder", "").replace("complete", "").strip()
        return None, content

    # 3. COMPLETE
    if clean.startswith("mark reminder") or clean.startswith("complete"):
        prefix = "mark reminder" if clean.startswith("mark reminder") else "complete"
        idx, title = _extract_target(clean, prefix)
        
        success = storage.complete_reminder(index=idx, partial_title=title)
        if success:
            logger.debug(f"  ⏰  [Reminders] Completed {idx or title}")
            return "REMINDER_COMPLETE", "Reminder marked complete."
        else:
            return "REMINDER_COMPLETE", "I couldn't find that reminder to complete."
            
    # 4. DELETE
    if clean.startswith("delete reminder") or clean.startswith("remove"):
        prefix = "delete reminder" if clean.startswith("delete reminder") else "remove"
        idx, title = _extract_target(clean, prefix)
        
        success = storage.delete_reminder(index=idx, partial_title=title)
        if success:
            logger.debug(f"  ⏰  [Reminders] Deleted {idx or title}")
            return "REMINDER_DELETE", "Reminder deleted."
        else:
            return "REMINDER_DELETE", "I couldn't find that reminder to delete."

    # 5. HISTORY
    from datetime import datetime
    
    def _format_time(iso_str: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_str)
            return dt.strftime("%I:%M %p").lstrip("0")
        except:
            return ""

    if clean in ("what reminders fired today", "what reminders fired today?"):
        history = storage.get_history()
        today_str = datetime.now().date().isoformat()
        fired_today = [h for h in history if h.get("fired_at", "").startswith(today_str)]
        if not fired_today:
            return "HISTORY_QUERY", "No reminders fired today."
        
        lines = []
        for h in fired_today:
            t = _format_time(h["fired_at"])
            lines.append(f"{h['title']} — {t}")
        return "HISTORY_QUERY", "\n".join(lines)

    if clean in ("show reminder history", "show reminder history."):
        history = storage.get_history()
        if not history:
            return "HISTORY_QUERY", "Reminder history is empty."
            
        lines = []
        for h in history[:5]: # Default to last 5
            t = _format_time(h["fired_at"])
            lines.append(f"{h['title']} — {t}")
        if len(history) > 5:
            lines.append(f"...and {len(history)-5} more.")
        return "HISTORY_QUERY", "\n".join(lines)
        
    if clean in ("how many reminders fired today", "how many reminders fired today?"):
        history = storage.get_history()
        today_str = datetime.now().date().isoformat()
        count = sum(1 for h in history if h.get("fired_at", "").startswith(today_str))
        return "HISTORY_QUERY", f"{count} reminders fired today."
        
    if clean.startswith("when did you remind me to"):
        task = clean.replace("when did you remind me to", "").strip().strip("?")
        history = storage.get_history()
        for h in history:
            if task in h["title"].lower():
                t = _format_time(h["fired_at"])
                return "HISTORY_QUERY", f"I reminded you to {h['title']} at {t}."
        return "HISTORY_QUERY", f"I haven't reminded you to {task} recently."

    return None, None
