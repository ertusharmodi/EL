"""
LLM-based memory extraction — natural language understanding.

Used as the second stage of the hybrid extraction pipeline after regex extraction.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

import ollama

import config

VALID_CATEGORIES = frozenset(
    {"personal", "preferences", "relationships", "goals", "projects", "skills", "facts"}
)

_EXTRACTION_SYSTEM_PROMPT = """You extract durable personal facts from user messages for a voice assistant memory system.

Return ONLY valid JSON. No markdown, no code fences, no explanation.

Schema:
{
  "memories": [
    {
      "category": "personal|preferences|relationships|goals|projects|skills|facts",
      "key": "snake_case_key",
      "value": "string OR array of strings for tech_stack",
      "confidence": 0.0
    }
  ]
}

Rules:
- Extract only lasting personal facts: preferences, relationships, goals, skills, birthplace, profession, etc.
- Ignore casual chitchat, questions, commands, greetings, and small talk with no personal value.
- Use snake_case keys. Use dot notation for nested keys (e.g. friend_rahul.location).
- For tech/skills, use category "skills", key "tech_stack", value as array.
- For relationships: sister, friend_<name>, friend_<name>.location, etc.
- confidence: float 0.0–1.0 — how certain this is a correct, lasting fact about the user.
- Return {"memories": []} when nothing worth saving.

Examples:
User: "My favourite color is black."
{"memories":[{"category":"preferences","key":"favorite_color","value":"black","confidence":0.98}]}

User: "I want to build a SaaS company in the future."
{"memories":[{"category":"goals","key":"business_goal","value":"build a SaaS company","confidence":0.92}]}

User: "My sister is a teacher."
{"memories":[{"category":"relationships","key":"sister","value":"teacher","confidence":0.95}]}

User: "My friend Rahul lives in Jaipur."
{"memories":[{"category":"relationships","key":"friend_rahul.location","value":"Jaipur","confidence":0.93}]}

User: "I prefer remote work."
{"memories":[{"category":"preferences","key":"work_style","value":"remote","confidence":0.9}]}

User: "I love Laravel and Node.js."
{"memories":[{"category":"skills","key":"tech_stack","value":["Laravel","Node.js"],"confidence":0.88}]}

User: "What's the weather like?"
{"memories":[]}"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_json_response(raw: str) -> Optional[dict]:
    """Extract and parse JSON from LLM output."""
    text = raw.strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    # Find outermost JSON object if surrounded by extra text.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_memory_item(item: dict) -> Optional[Dict[str, Any]]:
    """Validate and normalise a single memory dict from the LLM."""
    category = str(item.get("category", "")).strip().lower()
    key = str(item.get("key", "")).strip().lower()
    value = item.get("value")
    confidence = item.get("confidence")

    if category not in VALID_CATEGORIES or not key:
        return None

    if not isinstance(confidence, (int, float)):
        return None
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        return None

    if category == "skills" and key == "tech_stack":
        if isinstance(value, str):
            value = [v.strip() for v in re.split(r",|\band\b", value) if v.strip()]
        if not isinstance(value, list) or not value:
            return None
        value = [str(v).strip() for v in value if str(v).strip()]
    else:
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        if not isinstance(value, str) or not value.strip():
            return None
        value = value.strip()

    return {
        "category": category,
        "key": key,
        "value": value,
        "confidence": confidence,
        "source": "llm",
    }


def extract_memories_llm(
    user_message: str,
    already_extracted: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract memories from natural language using the LLM.

    already_extracted: regex results — passed as context so the LLM avoids duplicates.
    Returns an empty list on failure or when nothing is found.
    """
    clean = user_message.strip()
    if not clean or not config.MEMORY_LLM_EXTRACTION_ENABLED:
        return []

    context_block = ""
    if already_extracted:
        keys = [f"{m['category']}.{m['key']}" for m in already_extracted]
        context_block = (
            f"\n\nAlready extracted by regex (do not duplicate): {', '.join(keys)}"
        )

    t0 = time.monotonic()
    try:
        response = ollama.chat(
            model=config.MEMORY_LLM_MODEL,
            think=False,
            options={
                "num_predict": config.MEMORY_LLM_NUM_PREDICT,
                "temperature": config.MEMORY_LLM_TEMPERATURE,
            },
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": clean + context_block},
            ],
        )
    except Exception as exc:
        print(f"  ⚠️   [Memory LLM] Extraction failed: {exc}")
        return []

    elapsed = time.monotonic() - t0
    content = (response.message.content or "").strip()
    parsed = _parse_json_response(content)

    if parsed is None:
        print(f"  ⚠️   [Memory LLM] Could not parse JSON ({elapsed:.1f}s)")
        return []

    raw_memories = parsed.get("memories", [])
    if not isinstance(raw_memories, list):
        return []

    results: List[Dict[str, Any]] = []
    for item in raw_memories:
        if not isinstance(item, dict):
            continue
        normalised = _normalize_memory_item(item)
        if normalised:
            results.append(normalised)

    print(f"  🧠  [Memory LLM] {elapsed:.1f}s — {len(results)} candidate(s)")
    return results
