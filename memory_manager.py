import json
import os
from typing import Any, List, Literal, Union

import config

_memory_cache = None

_DEFAULT_STRUCTURE = {
    "personal": {},
    "preferences": {},
    "relationships": {},
    "goals": {},
    "projects": {},
    "facts": {},
    "skills": {},
}

SaveAction = Literal["saved", "updated", "unchanged", "skipped"]


def load_memory() -> None:
    """Load long_term.json into memory. Creates the file if it doesn't exist."""
    global _memory_cache
    path = config.MEMORY_LONG_TERM_FILE

    if not os.path.exists(path):
        _memory_cache = dict(_DEFAULT_STRUCTURE)
        save_memory()
    else:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            _memory_cache = dict(_DEFAULT_STRUCTURE)
            _memory_cache.update(loaded)
            for key in _DEFAULT_STRUCTURE:
                if key not in _memory_cache or not isinstance(_memory_cache[key], dict):
                    _memory_cache[key] = {}
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ⚠️   [Memory] Could not load long-term memory ({exc}) — starting fresh.")
            _memory_cache = dict(_DEFAULT_STRUCTURE)
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
        print(f"🧠 DEBUG: [8] Successfully wrote long_term.json to {path}")
    except OSError as exc:
        print(f"🧠 DEBUG: [8] Failed to write long_term.json: {exc}")
        print(f"  ⚠️   [Memory] Could not save long-term memory ({exc}).")


def dedupe_list(items: List[str]) -> List[str]:
    """Case-insensitive deduplication; preserve first-seen casing."""
    seen = set()
    result = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        fold = normalized.casefold()
        if fold not in seen:
            seen.add(fold)
            result.append(normalized)
    return result


def _ensure_loaded() -> dict:
    global _memory_cache
    if _memory_cache is None:
        load_memory()
    return _memory_cache


def _get_nested(cat_dict: dict, key_path: str) -> Any:
    """Read a value using dot notation (e.g. friend_rahul.location)."""
    if "." not in key_path:
        return cat_dict.get(key_path)

    cur: Any = cat_dict
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_nested(cat_dict: dict, key_path: str, value: Any) -> None:
    """Write a value using dot notation, creating intermediate dicts as needed."""
    parts = key_path.split(".")
    if len(parts) == 1:
        cat_dict[parts[0]] = value
        return

    cur = cat_dict
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def remember(category: str, key: str, value: Union[str, List[str], dict]) -> None:
    """Store a fact under a category and persist to disk."""
    cache = _ensure_loaded()
    if category not in cache:
        cache[category] = {}
    _set_nested(cache[category], key, value)
    save_memory()


def merge_tech_stack(new_items: List[str]) -> bool:
    """Merge new tech items into skills.tech_stack. Returns True if changed."""
    cache = _ensure_loaded()
    existing = get_value("skills", "tech_stack")
    if not isinstance(existing, list):
        existing = []

    merged = dedupe_list(existing + new_items)
    if merged == existing:
        return False

    if "skills" not in cache:
        cache["skills"] = {}
    cache["skills"]["tech_stack"] = merged
    save_memory()
    return True


def get_value(category: str, key: str) -> Any:
    """Return a stored value, or None if missing. Supports dot-notation keys."""
    cache = _ensure_loaded()
    cat_dict = cache.get(category, {})
    if not isinstance(cat_dict, dict):
        return None
    return _get_nested(cat_dict, key)


def is_list_like_key(category: str, key: str) -> bool:
    if category in ("skills", "interests", "preferences"):
        return True
    if key.startswith("favorite_"):
        return True
    return False


def merge_values(category: str, key: str, old_val: Any, new_val: Any) -> Any:
    def _to_list(v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(i).strip() for i in v]
        if isinstance(v, str):
            import re
            parts = re.split(r",|\band\b", v, flags=re.IGNORECASE)
            return [p.strip() for p in parts if p.strip()]
        return [str(v)]

    if is_list_like_key(category, key):
        old_list = _to_list(old_val) if old_val else []
        new_list = _to_list(new_val) if new_val else []
        merged = dedupe_list(old_list + new_list)
        if len(merged) == 1:
            return merged[0]
        elif len(merged) == 0:
            return None
        return merged
    else:
        old_str = str(old_val) if old_val else ""
        new_str = str(new_val) if new_val else ""
        if old_str.lower() == new_str.lower():
            return old_val
            
        if old_str.lower() in new_str.lower():
            return new_val
        elif new_str.lower() in old_str.lower():
            return old_val
            
        if len(new_str) > len(old_str):
            return new_val
        return old_val


def apply_memory(
    category: str,
    key: str,
    value: Union[str, List[str]],
) -> SaveAction:
    """
    Save or update a memory item. Returns:
      - 'saved'     — new key
      - 'updated'   — existing key, value changed
      - 'unchanged' — same value already stored
    """
    existing = get_value(category, key)
    
    if existing is None:
        remember(category, key, value)
        return "saved"
        
    merged = merge_values(category, key, existing, value)
    
    def _norm(v):
        if isinstance(v, list):
            return sorted([str(i).casefold() for i in v])
        return str(v).casefold() if v is not None else ""
        
    if _norm(existing) == _norm(merged):
        return "unchanged"
        
    print(f"  🧠 Merge Decision")
    print(f"  Old Value: {existing}")
    print(f"  New Value: {value}")
    print(f"  Final Value: {merged}")
    
    remember(category, key, merged)
    return "updated"


def forget(category: str, key: str) -> None:
    """Remove a fact from a category and persist to disk."""
    cache = _ensure_loaded()
    if category not in cache:
        return

    cat_dict = cache[category]
    if "." not in key:
        cat_dict.pop(key, None)
        save_memory()
        return

    parts = key.split(".")
    cur = cat_dict
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)
        save_memory()


def recall() -> dict:
    """Return the entire long-term memory dictionary."""
    return _ensure_loaded()


def _format_nested(prefix: str, value: Any, lines: List[str]) -> int:
    """Recursively format nested dict values for prompt injection."""
    count = 0
    if isinstance(value, dict):
        for k, v in value.items():
            nested_key = f"{prefix}.{k}" if prefix else k
            count += _format_nested(nested_key, v, lines)
    elif isinstance(value, list):
        lines.append(f"- {prefix}: {', '.join(str(i) for i in value)}")
        count += 1
    else:
        lines.append(f"- {prefix}: {value}")
        count += 1
    return count


def build_memory_context() -> str:
    """Convert the JSON memory into clean natural language context."""
    mem = recall()
    lines: List[str] = []
    fact_count = 0

    for cat, facts in mem.items():
        if not facts:
            continue
        lines.append(f"\n{cat.capitalize()}:")
        for k, v in facts.items():
            if isinstance(v, dict):
                fact_count += _format_nested(k, v, lines)
            elif isinstance(v, list):
                lines.append(f"- {k}: {', '.join(str(item) for item in v)}")
                fact_count += 1
            else:
                lines.append(f"- {k}: {v}")
                fact_count += 1

    if fact_count == 0:
        return ""

    print(f"  🧠  [Memory] Injected {fact_count} facts into prompt")

    context_str = "Known facts about the user:\n" + "\n".join(lines)
    return (
        "\n\n── LONG-TERM MEMORY ──────────────────────────────────────────────────────────────────\n"
        + context_str
    )
