import json
import logger
import os
import uuid
import threading
from typing import List, Dict, Any
from datetime import datetime

REMINDERS_FILE = "reminders/reminders.json"
HISTORY_FILE = "reminders/history.json"
_LOCK = threading.Lock()

def _ensure_dir():
    os.makedirs(os.path.dirname(REMINDERS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

_reminders_cache = None

def load_reminders() -> List[Dict[str, Any]]:
    """Loads reminders from JSON and caches them in memory."""
    global _reminders_cache
    with _LOCK:
        if _reminders_cache is not None:
            return list(_reminders_cache)
            
        _ensure_dir()
        if not os.path.exists(REMINDERS_FILE):
            _reminders_cache = []
            return []
            
        try:
            with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _reminders_cache = data.get("reminders", [])
                return list(_reminders_cache)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"  ⚠️  [Reminders] Failed to load reminders: {e}")
            _reminders_cache = []
            return []

def get_history() -> List[Dict[str, Any]]:
    """Loads reminder history sorted by fired_at descending."""
    with _LOCK:
        _ensure_dir()
        if not os.path.exists(HISTORY_FILE):
            return []
            
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                history = data.get("history", [])
                # Sort descending by fired_at
                return sorted(history, key=lambda x: x.get("fired_at", ""), reverse=True)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"  ⚠️  [Reminders] Failed to load history: {e}")
            return []

def add_to_history(reminder: Dict[str, Any], status: str) -> None:
    """Appends a reminder to the history log."""
    history = get_history()
    entry = {
        "id": reminder["id"],
        "title": reminder["title"],
        "created_at": reminder["created_at"],
        "due_at": reminder["due_at"],
        "fired_at": datetime.now().isoformat(),
        "status": status
    }
    history.append(entry)
    
    with _LOCK:
        _ensure_dir()
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump({"history": history}, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning(f"  ⚠️  [Reminders] Failed to save history: {e}")


def save_reminders(reminders: List[Dict[str, Any]]) -> None:
    """Saves reminders to JSON."""
    global _reminders_cache
    with _LOCK:
        _ensure_dir()
        data = {"reminders": reminders}
        try:
            with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            _reminders_cache = list(reminders)
        except OSError as e:
            logger.warning(f"  ⚠️  [Reminders] Failed to save reminders: {e}")

def create_reminder(title: str, due_at: str, recurring: bool = False, recurrence_rule: str = None) -> str:
    """Creates a new reminder and returns its UUID."""
    reminders = load_reminders()
    new_id = str(uuid.uuid4())
    reminders.append({
        "id": new_id,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "due_at": due_at,
        "completed": False,
        "recurring": recurring,
        "recurrence_rule": recurrence_rule,
        "paused": False
    })
    save_reminders(reminders)
    return new_id

def update_reminder(reminder_id: str, updates: Dict[str, Any]) -> bool:
    """Updates fields of an existing reminder. Returns True if found."""
    reminders = load_reminders()
    found = False
    for r in reminders:
        if r["id"] == reminder_id:
            r.update(updates)
            found = True
            break
    if found:
        save_reminders(reminders)
    return found

def list_reminders(include_completed: bool = False) -> List[Dict[str, Any]]:
    """Returns the list of active reminders."""
    reminders = load_reminders()
    if include_completed:
        return reminders
    return [r for r in reminders if not r.get("completed", False)]

def complete_reminder(index: int = None, partial_title: str = None) -> bool:
    """
    Marks a reminder complete by 1-based index (of active ones) or substring match.
    Returns True if found and completed.
    """
    reminders = load_reminders()
    active = [r for r in reminders if not r.get("completed", False)]
    
    target_id = None
    if index is not None and 1 <= index <= len(active):
        target_id = active[index - 1]["id"]
    elif partial_title:
        for r in active:
            if partial_title.lower() in r["title"].lower():
                target_id = r["id"]
                break
                
    if not target_id:
        return False
        
    for r in reminders:
        if r["id"] == target_id:
            r["completed"] = True
            save_reminders(reminders)
            return True
            
    return False

def delete_reminder(index: int = None, partial_title: str = None) -> bool:
    """
    Deletes a reminder completely by 1-based index or substring match.
    Returns True if found and deleted.
    """
    reminders = load_reminders()
    active = [r for r in reminders if not r.get("completed", False)]
    
    target_id = None
    if index is not None and 1 <= index <= len(active):
        target_id = active[index - 1]["id"]
    elif partial_title:
        for r in active:
            if partial_title.lower() in r["title"].lower():
                target_id = r["id"]
                break
                
    if not target_id:
        return False
        
    reminders = [r for r in reminders if r["id"] != target_id]
    save_reminders(reminders)
    return True
