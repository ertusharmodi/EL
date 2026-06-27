# memory.py — Memory layer for Eleven.
#
# Responsibilities:
#   - Load short_term.json from disk on import (or start fresh if missing).
#   - Load profile.json from disk on import (or return {} if missing).
#   - add_turn(user, assistant) — append a turn, trim to the cap, save.
#   - get_history()             — return Ollama-ready message dicts.
#   - get_profile()             — return user profile dict for prompt injection.
#   - clear()                   — wipe in-memory + on-disk history.
#
# JSON schema (memory/short_term.json):
# {
#   "turns": [
#     {"role": "user",      "content": "..."},
#     {"role": "assistant", "content": "..."},
#     ...
#   ]
# }
#
# Each conversation turn produces 2 entries (user + assistant).
# The list is capped at MEMORY_SHORT_TERM_TURNS * 2 entries.

import json
import os
import time
import inspect
from typing import List

import config
import logger

# ── Internal state ─────────────────────────────────────────────────────────────
# List of {"role": "user"|"assistant", "content": "..."} dicts.
# Loaded from disk at import; kept in memory for the session lifetime.
_history: List[dict] = []

# User profile dict loaded from profile.json at import.
# Keys are arbitrary (name, city, profession, …); values are strings.
_profile: dict = {}

_last_save_time: float = 0.0


def _load() -> None:
    """Load short_term.json into _history. Silent no-op if file is missing."""
    global _history
    path = config.MEMORY_SHORT_TERM_FILE
    if not os.path.exists(path):
        _history = []
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        _history = data.get("turns", [])
        logger.debug(f"  🧠  [MEMORY] loaded recent turns ({len(_history) // 2})")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"  ⚠️   [MEMORY] Could not load memory ({exc}) — starting fresh.")
        _history = []


def _save() -> None:
    """Persist _history to short_term.json."""
    os.makedirs(config.MEMORY_DIR, exist_ok=True)
    path = config.MEMORY_SHORT_TERM_FILE
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"turns": _history}, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning(f"  ⚠️   [MEMORY] Could not save memory ({exc}).")


def _trim() -> None:
    """Keep only the last MEMORY_SHORT_TERM_TURNS turns (2 entries per turn)."""
    global _history
    max_entries = config.MEMORY_SHORT_TERM_TURNS * 2
    if len(_history) > max_entries:
        _history = _history[-max_entries:]


def _load_profile() -> None:
    """Load profile.json into _profile. Silent no-op if file is missing."""
    global _profile
    path = config.MEMORY_PROFILE_FILE
    if not os.path.exists(path):
        _profile = {}
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            _profile = json.load(fh)
        logger.debug(f"  🧠  [MEMORY] loaded profile ({len(_profile)} fields)")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"  ⚠️   [MEMORY] Could not load profile ({exc}) — using empty profile.")
        _profile = {}


def _save_profile() -> None:
    """Persist _profile to profile.json."""
    os.makedirs(config.MEMORY_DIR, exist_ok=True)
    path = config.MEMORY_PROFILE_FILE
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_profile, fh, ensure_ascii=False, indent=2)
        logger.debug("  🧠  [MEMORY] saved profile field")
    except OSError as exc:
        logger.warning(f"  ⚠️   [MEMORY] Could not save profile ({exc}).")


# ── Public API ─────────────────────────────────────────────────────────────────

def get_profile() -> dict:
    """
    Return the user profile as a flat dict.

    Safe to render into the system prompt:
        for key, value in memory.get_profile().items(): ...
    """
    return dict(_profile)


def set_profile_fields(fields: dict) -> None:
    """
    Merge the given fields into the user profile and persist to disk.
    Only fields that are strings and actually changed will be saved.
    """
    global _profile
    changed = False
    for k, v in fields.items():
        # Only accept basic string facts.
        if isinstance(v, str) and _profile.get(k) != v:
            _profile[k] = v
            changed = True
            
    if changed:
        _save_profile()


def get_history() -> List[dict]:
    """
    Return the stored conversation turns as a list of Ollama message dicts.

    Safe to inject directly into the messages list:
        messages = [system_msg, *memory.get_history(), user_msg]
    """
    return list(_history)


def add_turn(user_text: str, assistant_text: str, response_time_ms: int = None) -> None:
    """
    Append one conversation turn and persist to disk.

    Args:
        user_text:      The user's message for this turn.
        assistant_text: Eleven's response for this turn.
        response_time_ms: Response time in milliseconds.
    """
    global _history, _last_save_time
    import conversation
    
    current_time = time.time()
    user_ts = conversation.get_current_timestamp()
    
    # We add 2 seconds arbitrarily to assistant_ts if they are the same in testing, 
    # but in real usage they will be virtually identical.
    # To be perfectly accurate to the request, we just take the current timestamp again.
    # Actually, the user wants us to log when they are saved.
    assistant_ts = conversation.get_current_timestamp()
    
    # ── Deduplication Check ──────────────────────────────────────────────────
    if _history:
        last_turn = _history[-1]
        time_diff = current_time - _last_save_time
        
        # Check if we're trying to add the exact same assistant turn within 5 seconds
        if time_diff <= 5.0 and last_turn.get("role") == "assistant" and last_turn.get("content") == assistant_text:
            logger.debug("  ⚠️  Duplicate turn skipped")
            return

    # Get caller info
    try:
        caller_frame = inspect.currentframe().f_back
        caller_file = os.path.basename(caller_frame.f_code.co_filename)
        caller_func = caller_frame.f_code.co_name
    except Exception:
        caller_file = "unknown"
        caller_func = "unknown"
        
    logger.debug(f"  [MEMORY SAVE] User turn added (Function: {caller_func}, File: {caller_file})")
    logger.debug(f"  [CHAT] User message saved @ {user_ts}")
    _history.append({"role": "user", "content": user_text, "timestamp": user_ts})
    
    logger.debug(f"  [MEMORY SAVE] Assistant turn added (Function: {caller_func}, File: {caller_file})")
    logger.debug(f"  [CHAT] Assistant response saved @ {assistant_ts}")
    
    assistant_msg = {"role": "assistant", "content": assistant_text, "timestamp": assistant_ts}
    if response_time_ms is not None:
        assistant_msg["response_time_ms"] = response_time_ms
        
    _history.append(assistant_msg)
    
    _last_save_time = current_time
    _trim()
    _save()


def clear() -> None:
    """Wipe all short-term memory, in memory and on disk."""
    global _history
    _history = []
    _save()
    logger.debug("  🧠  [MEMORY] Cleared.")


def turn_count() -> int:
    """Return the number of stored turns (not entries)."""
    return len(_history) // 2


# ── Load on import ─────────────────────────────────────────────────────────────
_load()
_load_profile()
