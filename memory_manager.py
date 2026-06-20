import json
import os

import config

_memory_cache = None

def load_memory() -> None:
    """Load long_term.json into memory. Creates the file if it doesn't exist."""
    global _memory_cache
    path = config.MEMORY_LONG_TERM_FILE
    
    default_structure = {
        "personal": {},
        "preferences": {},
        "relationships": {},
        "goals": {},
        "projects": {},
        "facts": {}
    }

    if not os.path.exists(path):
        _memory_cache = default_structure
        save_memory()
    else:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                _memory_cache = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ⚠️   [Memory] Could not load long-term memory ({exc}) — starting fresh.")
            _memory_cache = default_structure
            save_memory()
            
    print(f"  🧠  [Memory] Loaded {len(_memory_cache)} categories")


def save_memory() -> None:
    """Persist the memory cache to long_term.json."""
    if _memory_cache is None:
        return
        
    os.makedirs(config.MEMORY_DIR, exist_ok=True)
    path = config.MEMORY_LONG_TERM_FILE
    
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_memory_cache, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"  ⚠️   [Memory] Could not save long-term memory ({exc}).")


def remember(category: str, key: str, value: str) -> None:
    """Store a specific fact under a category and persist to disk."""
    global _memory_cache
    if _memory_cache is None:
        load_memory()
        
    if category not in _memory_cache:
        _memory_cache[category] = {}
        
    _memory_cache[category][key] = value
    save_memory()


def forget(category: str, key: str) -> None:
    """Remove a fact from a category and persist to disk."""
    global _memory_cache
    if _memory_cache is None:
        load_memory()
        
    if category in _memory_cache and key in _memory_cache[category]:
        del _memory_cache[category][key]
        save_memory()


def recall() -> dict:
    """Return the entire long-term memory dictionary."""
    if _memory_cache is None:
        load_memory()
    return _memory_cache


def build_memory_context() -> str:
    """Convert the JSON memory into clean natural language context."""
    mem = recall()
    lines = []
    fact_count = 0
    
    for cat, facts in mem.items():
        if facts:
            lines.append(f"\n{cat.capitalize()}:")
            for k, v in facts.items():
                lines.append(f"- {k}: {v}")
                fact_count += 1
                
    if fact_count == 0:
        return ""
        
    print(f"  🧠  [Memory] Injected {fact_count} facts into prompt")
    
    context_str = "Known facts about the user:\n" + "\n".join(lines)
    return "\n\n── LONG-TERM MEMORY ──────────────────────────────────────────────────────────────────\n" + context_str
