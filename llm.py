# llm.py — Chat with qwen3 via Ollama, optimised for low-latency voice responses.
#
# qwen3 is a reasoning model: by default it generates a hidden chain-of-thought
# that adds 15–20 s of latency per turn.  We suppress it three ways:
#   1. think=False  — Ollama API flag (requires Ollama ≥ 0.9.0 / 0.30.x)
#   2. /no_think    — appended to system prompt as model-level instruction
#   3. num_predict  — hard token cap so CoT can't run away even if (1) is ignored
#   4. Strip        — regex removes any residual <think>...</think> from content

import os
import re
import time

import ollama

import config

# ── System prompt ────────────────────────────────────────────────────────────
_PERSONALITY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "personality.txt",
)

try:
    with open(_PERSONALITY_FILE, "r", encoding="utf-8") as _f:
        _SYSTEM_PROMPT = _f.read().strip()
except FileNotFoundError:
    _SYSTEM_PROMPT = config.OLLAMA_SYSTEM_PROMPT

# Append /no_think as a belt-and-suspenders measure.
# This is a model-level instruction that works even when the Ollama server
# does not propagate the think=False API flag correctly.
if not _SYSTEM_PROMPT.endswith("/no_think"):
    _SYSTEM_PROMPT = _SYSTEM_PROMPT + "\n\n/no_think"

# ── Response stripper ────────────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove any <think>...</think> blocks left in the content field."""
    return _THINK_RE.sub("", text).strip()


# ── Chat ─────────────────────────────────────────────────────────────────────

def chat(user_text: str) -> str:
    """
    Send user_text to the LLM and return the AI's response as a string.

    Uses a non-streaming, single-turn call (no conversation history).
    Chain-of-thought reasoning is suppressed via three mechanisms; see module
    docstring for details.

    Prints timing diagnostics: wall time, tokens generated, tokens/sec.
    """
    t0 = time.monotonic()

    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        think=config.OLLAMA_THINK,          # API-level suppression
        options={
            "num_predict": config.OLLAMA_NUM_PREDICT,   # hard token cap
        },
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
    )

    elapsed = time.monotonic() - t0

    # Strip residual <think> blocks (safety net for older Ollama builds)
    content = _strip_think(response.message.content)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    try:
        eval_tokens = response.eval_count
        eval_secs   = response.eval_duration / 1e9
        prompt_tok  = response.prompt_eval_count
        tps         = eval_tokens / eval_secs if eval_secs > 0 else 0
        print(
            f"  🧠  [LLM] {elapsed:.1f}s wall  |  "
            f"prompt={prompt_tok}tok  gen={eval_tokens}tok  "
            f"@ {tps:.0f}tok/s"
        )
    except Exception:
        print(f"  🧠  [LLM] {elapsed:.1f}s wall")

    return content
