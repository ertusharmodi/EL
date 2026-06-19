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
from typing import List

import config

# ── Internal state ─────────────────────────────────────────────────────────────
# List of {"role": "user"|"assistant", "content": "..."} dicts.
# Loaded from disk at import; kept in memory for the session lifetime.
_history: List[dict] = []

# User profile dict loaded from profile.json at import.
# Keys are arbitrary (name, city, profession, …); values are strings.
_profile: dict = {}


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
        print(f"  🧠  [Memory] Loaded {len(_history) // 2} turn(s) from {path}")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ⚠️   [Memory] Could not load memory ({exc}) — starting fresh.")
        _history = []


def _save() -> None:
    """Persist _history to short_term.json."""
    os.makedirs(config.MEMORY_DIR, exist_ok=True)
    path = config.MEMORY_SHORT_TERM_FILE
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"turns": _history}, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"  ⚠️   [Memory] Could not save memory ({exc}).")


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
        print(f"  🧠  [Memory] Loaded profile ({len(_profile)} field(s)) from {path}")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ⚠️   [Memory] Could not load profile ({exc}) — using empty profile.")
        _profile = {}


# ── Public API ─────────────────────────────────────────────────────────────────

def get_profile() -> dict:
    """
    Return the user profile as a flat dict.

    Safe to render into the system prompt:
        for key, value in memory.get_profile().items(): ...
    """
    return dict(_profile)


def get_history() -> List[dict]:
    """
    Return the stored conversation turns as a list of Ollama message dicts.

    Safe to inject directly into the messages list:
        messages = [system_msg, *memory.get_history(), user_msg]
    """
    return list(_history)


def add_turn(user_text: str, assistant_text: str) -> None:
    """
    Append one conversation turn and persist to disk.

    Args:
        user_text:      The user's message for this turn.
        assistant_text: Eleven's response for this turn.
    """
    _history.append({"role": "user",      "content": user_text})
    _history.append({"role": "assistant", "content": assistant_text})
    _trim()
    _save()


def clear() -> None:
    """Wipe all short-term memory, in memory and on disk."""
    global _history
    _history = []
    _save()
    print("  🧠  [Memory] Cleared.")


def turn_count() -> int:
    """Return the number of stored turns (not entries)."""
    return len(_history) // 2


# ── Load on import ─────────────────────────────────────────────────────────────
_load()
_load_profile()
