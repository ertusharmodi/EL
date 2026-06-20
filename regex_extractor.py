"""
Regex-based memory extraction — fast, deterministic pattern matching.

Used as the first stage of the hybrid extraction pipeline.
"""

import re
from typing import Any, Dict, List

import memory_manager

# ── Regex patterns ────────────────────────────────────────────────────────────

_RE_FAVORITE = re.compile(
    r"my favo(?:u)?rite (.+?) is (.+?)(?=\s+and\s|\.\s|\.$|$)",
    re.IGNORECASE,
)

_RE_BIRTHPLACE_EXPLICIT = re.compile(
    r"(?:my birthplace is|i was born in)\s+(.+?)(?=\s+and\s|\.\s|\.$|$)",
    re.IGNORECASE,
)

_RE_BIRTHPLACE_FROM = re.compile(
    r"(?:^|\s)(?:i'm from|i am from)\s+(.+?)(?=\s+and\s|\.\s|\.$|$)",
    re.IGNORECASE,
)

_RE_PROFESSION_COMPOUND = re.compile(
    r"(?:i am a|i'm a|i work as an?)\s+(.+?)\s+from\s+(.+?)(?=\s+and\s|\.\s|\.$|$)",
    re.IGNORECASE,
)

_RE_PROFESSION = re.compile(
    r"(?:i am a|i'm a|i work as an?)\s+(.+?)(?=\s+from\s|\s+in\s|\s+and\s|\.\s|\.$|$)",
    re.IGNORECASE,
)

_RE_SKILLS = re.compile(
    r"(?:i use|i mostly work with|i work with|i have experience with|i love)\s+(.+?)(?=\s+and\s+i\s|\s+from\s|\.\s|\.$|$)",
    re.IGNORECASE,
)

_RE_LOCATION = re.compile(
    r"i live in\s+(.+?)(?=\s+and\s|\.\s|\.$|$)",
    re.IGNORECASE,
)

_RE_EXPERIENCE = re.compile(
    r"i have\s+(.+?)\s+of experience(?:\s|$|\.)",
    re.IGNORECASE,
)

_RE_GOAL = re.compile(
    r"my goal is to\s+(.+?)(?=\s+and\s|\.\s|\.$|$)",
    re.IGNORECASE,
)

_RE_PROJECT = re.compile(
    r"my main project is\s+(.+?)(?=\s+and\s|\.\s|\.$|$)",
    re.IGNORECASE,
)

# Regex matches are treated as fully confident.
REGEX_CONFIDENCE = 1.0


def _clean_message(user_message: str) -> str:
    return user_message.strip().strip(".!? ").strip()


def _clean_value(value: str) -> str:
    return value.strip().strip(",.;:!?")


def _title_case_profession(value: str) -> str:
    return _clean_value(value).title()


def _favorite_key(thing: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", thing.strip().lower()).strip("_")
    return f"favorite_{slug}"


def _parse_tech_list(text: str) -> List[str]:
    parts = re.split(r",|\band\b", text, flags=re.IGNORECASE)
    return [_clean_value(part) for part in parts if _clean_value(part)]


def _dedupe_memories(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scalar: Dict[tuple, Dict[str, Any]] = {}
    tech_items: List[str] = []

    for mem in memories:
        cat, key, val = mem["category"], mem["key"], mem["value"]
        if cat == "skills" and key == "tech_stack":
            tech_items.extend(val)
        else:
            scalar[(cat, key)] = mem

    result = list(scalar.values())
    if tech_items:
        result.append(
            {
                "category": "skills",
                "key": "tech_stack",
                "value": memory_manager.dedupe_list(tech_items),
                "confidence": REGEX_CONFIDENCE,
                "source": "regex",
            }
        )
    return result


def extract_memories_regex(user_message: str) -> List[Dict[str, Any]]:
    """
    Extract structured memories using regex rules.
    Each item includes confidence=1.0 and source='regex'.
    """
    clean_msg = _clean_message(user_message)
    if not clean_msg:
        return []

    memories: List[Dict[str, Any]] = []

    def _add(category: str, key: str, value: Any) -> None:
        memories.append(
            {
                "category": category,
                "key": key,
                "value": value,
                "confidence": REGEX_CONFIDENCE,
                "source": "regex",
            }
        )

    for match in _RE_FAVORITE.finditer(clean_msg):
        thing, value = match.group(1), match.group(2)
        _add("preferences", _favorite_key(thing), _clean_value(value))

    for match in _RE_BIRTHPLACE_EXPLICIT.finditer(clean_msg):
        _add("personal", "birthplace", _clean_value(match.group(1)))

    for match in _RE_BIRTHPLACE_FROM.finditer(clean_msg):
        _add("personal", "birthplace", _clean_value(match.group(1)))

    for match in _RE_PROFESSION_COMPOUND.finditer(clean_msg):
        profession, birthplace = match.group(1), match.group(2)
        _add("personal", "profession", _title_case_profession(profession))
        _add("personal", "birthplace", _clean_value(birthplace))

    for match in _RE_PROFESSION.finditer(clean_msg):
        _add("personal", "profession", _title_case_profession(match.group(1)))

    for match in _RE_SKILLS.finditer(clean_msg):
        techs = _parse_tech_list(match.group(1))
        if techs:
            _add("skills", "tech_stack", techs)

    for match in _RE_LOCATION.finditer(clean_msg):
        _add("personal", "location", _clean_value(match.group(1)))

    for match in _RE_EXPERIENCE.finditer(clean_msg):
        _add("personal", "experience", _clean_value(match.group(1)))

    for match in _RE_GOAL.finditer(clean_msg):
        _add("goals", "primary_goal", _clean_value(match.group(1)))

    for match in _RE_PROJECT.finditer(clean_msg):
        _add("projects", "main_project", _clean_value(match.group(1)))

    return _dedupe_memories(memories)
