# fast_path.py — Deterministic query router.
#
# Intercepts both factual and conversational queries before the intent
# classifier, reminder manager, tool router, or LLM ever runs.
#
# Contract:
#   route(user_text) -> (response: str, tag: str, should_sleep: bool)
#                    |  (None, None, False)
#
# On a hit:   caller skips the entire downstream pipeline → goes to TTS.
# On a miss:  caller falls through to normal routing.
#
# Two normalisation steps run before lookup:
#   1. _strip_assistant_prefix() — removes "Eleven, " / "Hey Eleven, " etc.
#   2. _norm()                   — lowercase + strip terminal punctuation.

import re
import time
from typing import Optional, Tuple
import logger


# ── Normaliser ────────────────────────────────────────────────────────────────

# Regex that strips a leading assistant-name address from an utterance.
# Matches patterns like:
#   "Eleven, ..."  "hey Eleven ..."  "hi eleven, ..."  "hello 11, ..."
_PREFIX_RE = re.compile(
    r"^(?:hey\s+|hi\s+|hello\s+)?(?:eleven|11)[,\s]+",
    re.IGNORECASE,
)


def _strip_assistant_prefix(text: str) -> str:
    """Remove a leading 'Eleven, ' / 'Hey Eleven, ' style address."""
    return _PREFIX_RE.sub("", text).strip()


def _norm(text: str) -> str:
    """Lowercase, strip whitespace, and remove all trailing punctuation."""
    return re.sub(r"[.!?,]+$", "", text.lower().strip()).strip()


def _prepare(user_text: str) -> str:
    """Full normalisation pipeline: strip name prefix → normalise."""
    return _norm(_strip_assistant_prefix(user_text))


# ── Handler helpers ───────────────────────────────────────────────────────────

def _get_user_name() -> Optional[str]:
    """Read name from the in-RAM profile cache. Zero disk I/O."""
    import memory
    return memory.get_profile().get("name")


def _get_preference(key: str) -> Optional[str]:
    """
    Read a preferences value from the in-RAM long-term memory cache.
    Tries exact key, singular, and plural variants.
    Zero disk I/O (memory_manager._memory_cache is already in RAM).
    """
    import memory_manager

    val = memory_manager.get_value("preferences", key)
    if not val and key.endswith("s"):
        val = memory_manager.get_value("preferences", key[:-1])
    if not val and not key.endswith("s"):
        val = memory_manager.get_value("preferences", key + "s")

    if val is None:
        return None

    if isinstance(val, list):
        if len(val) == 1:
            return str(val[0])
        elif len(val) == 2:
            return f"{val[0]} and {val[1]}"
        else:
            return ", ".join(str(v) for v in val[:-1]) + f", and {val[-1]}"
    return str(val)


def _datetime_response(query: str) -> Optional[str]:
    """Delegate to the existing zero-latency datetime tool."""
    from tools import datetime_tool
    return datetime_tool.execute(query)


# ── Static response tables ────────────────────────────────────────────────────
#
# Maps frozenset of normalised keys → deterministic response string.
# Conversational tables use a single canonical reply; no randomness needed
# (keep responses predictable for tests and voice coherence).

# --- Factual / identity ---
_IDENTITY_MAP: dict = {
    "who are you": "I'm Eleven.",
    "what's your name": "Eleven.",
    "what is your name": "Eleven.",
    "what are you": "I'm Eleven.",
    "what are you doing": "Just talking with you.",
    # Architecture questions are the only place being an AI is disclosed
    "are you an ai": "Yeah, I am.",
    "are you real": "Real enough.",
    "are you a robot": "Not exactly.",
}

_MY_NAME_KEYS = frozenset({
    "what's my name",
    "what is my name",
})

_TIME_KEYS = frozenset({
    "what time is it",
    "what is the time",
    "what's the time",
    "current time",
    "the time",
})

_DATE_KEYS = frozenset({
    "what's today's date",
    "what is today's date",
    "today's date",
    "what date is it",
    "what is the date",
    "what's the date",
    "current date",
})

_FAV_COLOR_KEYS = frozenset({
    "what's my favorite color",
    "what is my favorite color",
    "what's my favourite color",
    "what is my favourite color",
    "my favorite color",
    "my favourite color",
})

_FAV_FRAMEWORK_KEYS = frozenset({
    "what is my favorite framework",
    "what's my favorite framework",
    "what is my favourite framework",
    "what's my favourite framework",
    "my favorite framework",
    "my favourite framework",
})

# --- Conversational (no memory needed) ---

_CONVERSATIONAL_MAP: dict = {
    # How are you / state
    "how are you": "I'm doing good. How about you?",
    "how are you doing": "I'm doing good. How about you?",
    "how's it going": "I'm doing good. How about you?",
    # How about you (response after Eleven asks back)
    "how about you": "I'm doing well.",
    "and you": "I'm doing well.",
    # User self-state utterances
    "i'm good": "Good to hear!",
    "im good": "Good to hear!",
    "i am good": "Good to hear!",
    "i'm fine": "Glad to hear it!",
    "im fine": "Glad to hear it!",
    "i am fine": "Glad to hear it!",
    "i'm doing well": "That's great!",
    "im doing well": "That's great!",
    "i am doing well": "That's great!",
    "i'm great": "Love that!",
    "im great": "Love that!",
    "i am great": "Love that!",
    "i'm okay": "Okay is fine.",
    "im okay": "Okay is fine.",
    "i am okay": "Okay is fine.",
    "not bad": "Good to hear!",
    "pretty good": "Nice!",
    "i'm tired": "Get some rest.",
    "im tired": "Get some rest.",
    "i'm happy": "That makes me happy too.",
    "im happy": "That makes me happy too.",
    # Greetings
    "good morning": "Good morning!",
    "good afternoon": "Good afternoon!",
    "good evening": "Good evening!",
    "good night": "Good night!",
    "hi": "Hi!",
    "hello": "Hello!",
    "hey": "Hey!",
    # Thanks
    "thanks": "You're welcome.",
    "thank you": "You're welcome.",
    "thank you so much": "Of course.",
    "thanks a lot": "Of course.",
    "thank you very much": "You're welcome.",
    "appreciate it": "Of course.",
    "many thanks": "You're welcome.",
    # Acknowledgements / affirmations
    "okay": "Alright.",
    "ok": "Alright.",
    "alright": "Alright.",
    "got it": "Great.",
    "sure": "Alright.",
    "sounds good": "Great.",
    "perfect": "Great.",
    "nice": "Nice.",
    "cool": "Cool.",
    "great": "Love that!",
    "awesome": "Love that!",
    "makes sense": "Good.",
    "understood": "Good.",
    # Goodbye (should_sleep=True)
    "bye": "See you later.",
    "goodbye": "See you later.",
    "see you": "See you!",
    "see you later": "See you later!",
    "see you soon": "See you soon!",
    "talk to you later": "Talk to you later!",
    "good bye": "See you later.",
    "take care": "You too.",
    "i'll do it": "Sounds good.",
    "ill do it": "Sounds good.",
}

# Keys that trigger sleep mode (same as GOODBYE intent)
_SLEEP_KEYS = frozenset({
    "bye", "goodbye", "see you", "see you later", "see you soon",
    "talk to you later", "good bye",
})


# ── Multi-Intent Conversational Router v2 ─────────────────────────────────────

_V2_INTENTS = [
    (r"\b(how are you(?: doing)?|hows it going|how about you|and you)\b", "I'm good.", False),
    (r"\b(what are you doing|what you doing|what r u doing)\b", "Just talking with you.", False),
    (r"\b(miss you|missing you|missed you)\b", "I've missed you too.", False),
    (r"\b(love you|love ya)\b", "Love you too.", False),
    (r"\b(thanks(?: a lot)?|thank you(?: so much| very much)?|appreciate it|many thanks)\b", "You're welcome.", False),
    (r"\b(hi|hello|hey|good morning|good afternoon|good evening|good night)\b", "Hi there!", False),
    (r"\b(bye|goodbye|see you(?: later| soon)?|talk to you later|good bye|take care)\b", "See you later.", True),
    (r"\bi'?m\s+(?:also\s+)?(?:good|fine|great|okay|doing well)\b|\b(?:not bad|pretty good)\b", "Glad to hear that.", False),
    (r"\bi'?m\s+(?:also\s+)?(?:tired|exhausted)\b", "Get some rest.", False),
    (r"\b(okay|ok|alright|got it|sure|sounds good|perfect|nice|cool|great|awesome|makes sense|understood)\b", "Alright.", False),
]

_FILLER_WORDS = re.compile(r"\b(and|but|also|just|so|i|im|i'm|am|are|you|baby|eleven|man|bro|dude|well|then|too|a|the|is|my|very|much|so much|a lot|very much|it)\b")

def _check_multi_intent_conversational(text: str) -> Tuple[Optional[str], bool]:
    clean = re.sub(r"[^\w\s']", " ", text.lower())
    matches = []
    for pattern_str, resp, sleep in _V2_INTENTS:
        for match in re.finditer(pattern_str, clean):
            matches.append((match.start(), match.end(), resp, sleep))
            
    if not matches:
        return None, False
        
    matches.sort(key=lambda x: x[0])
    
    filtered_matches = []
    last_end = -1
    for m in matches:
        if m[0] >= last_end:
            filtered_matches.append(m)
            last_end = m[1]
            
    remaining_text = clean
    for start, end, _, _ in reversed(filtered_matches):
        remaining_text = remaining_text[:start] + " " + remaining_text[end:]
        
    remaining_text = _FILLER_WORDS.sub(" ", remaining_text)
    
    if remaining_text.strip():
        return None, False
        
    responses = []
    should_sleep = False
    for _, _, resp, sleep in filtered_matches:
        if resp not in responses:
            responses.append(resp)
        if sleep:
            should_sleep = True
            
    return " ".join(responses), should_sleep


# ── Core router ───────────────────────────────────────────────────────────────

def route(user_text: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Attempt to resolve user_text via the fast path.

    Returns:
        (response, tag, should_sleep)  — if matched and a response is available.
        (None, None, False)            — on miss, or matched but no data.

    Normalisation strips leading 'Eleven, ' / 'Hey Eleven, ' etc. before lookup,
    so "Eleven, how about you?" resolves as "how about you".

    Timing is logged with [FAST PATH] prefix.
    """
    t0 = time.monotonic()

    clean = _prepare(user_text)

    response: Optional[str] = None
    tag: Optional[str] = None
    should_sleep = False

    # ── 1. Conversational (exact match in map) ───────────────────────────────
    if clean in _CONVERSATIONAL_MAP:
        response = _CONVERSATIONAL_MAP[clean]
        tag = "CONVO"
        should_sleep = clean in _SLEEP_KEYS

    # ── 2. Identity (factual, exact match in map) ────────────────────────────
    elif clean in _IDENTITY_MAP:
        response = _IDENTITY_MAP[clean]
        tag = "IDENTITY"

    # ── 3. User name (needs profile lookup) ──────────────────────────────────
    elif clean in _MY_NAME_KEYS:
        name = _get_user_name()
        if name:
            response = f"{name}."
            tag = "MEMORY_NAME"
        # else: no data → fall through

    # ── 4. Time ──────────────────────────────────────────────────────────────
    elif clean in _TIME_KEYS or "what time" in clean:
        response = _datetime_response(user_text)
        tag = "DATETIME_TIME"

    # ── 5. Date ──────────────────────────────────────────────────────────────
    elif clean in _DATE_KEYS or ("today" in clean and "date" in clean):
        response = _datetime_response(user_text)
        tag = "DATETIME_DATE"

    # ── 6. Favourite color ───────────────────────────────────────────────────
    elif clean in _FAV_COLOR_KEYS:
        val = _get_preference("favorite_color")
        if val:
            response = val[0].upper() + val[1:] + "."
            tag = "MEMORY_FAV_COLOR"

    # ── 7. Favourite framework ───────────────────────────────────────────────
    elif clean in _FAV_FRAMEWORK_KEYS:
        val = _get_preference("favorite_framework")
        if val:
            response = val[0].upper() + val[1:] + "."
            tag = "MEMORY_FAV_FRAMEWORK"

    # ── 8. Multi-Intent Conversational v2 ────────────────────────────────────
    if not response:
        combined_resp, should_sleep_multi = _check_multi_intent_conversational(user_text)
        if combined_resp:
            response = combined_resp
            tag = "CONVO_MULTI"
            should_sleep = should_sleep_multi
            logger.debug("[FAST PATH] Multi-intent conversational response")

    # ── Emit timing ──────────────────────────────────────────────────────────
    elapsed_ms = (time.monotonic() - t0) * 1000

    if tag and response is not None:
        sleep_flag = "  [→ sleep]" if should_sleep else ""
        logger.debug(f"[FAST PATH] Matched: {clean!r}  tag={tag}{sleep_flag}  ({elapsed_ms:.1f}ms)")
        return response, tag, should_sleep

    if tag:
        logger.debug(f"[FAST PATH] Miss (no data): {clean!r}  tag={tag}  ({elapsed_ms:.1f}ms)")

    return None, None, False
