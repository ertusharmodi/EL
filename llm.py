# llm.py — Chat with qwen3 via Ollama, optimised for low-latency voice responses.
#
# qwen3 is a reasoning/thinking model.  The correct way to get fast, clean output
# on Ollama 0.30.x is:
#
#   think=True  — routes the chain-of-thought into message.thinking (separate field)
#                 rather than embedding it in message.content.  This means content
#                 is always the clean, final answer.
#
#   num_predict — cap total tokens (thinking + answer).  qwen3:4b typically uses
#                 100–400 thinking tokens for simple conversational queries.
#                 OLLAMA_NUM_PREDICT = 1200 is enough for thinking + a 2-sentence answer.
#
# What does NOT work on Ollama 0.30.x:
#   think=False  — model ignores it; thinking is generated anyway and embedded inline
#                  in message.content (not routed to message.thinking), making it
#                  impossible to strip cleanly.
#
# We also keep a regex stripper as a safety net for edge cases.

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

# ── Response stripper ────────────────────────────────────────────────────────
# Safety net: if content somehow includes a leaked <think> block, strip it.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


# ── Chat ─────────────────────────────────────────────────────────────────────

def chat(user_text: str) -> str:
    """
    Send user_text to the LLM and return the AI's response as a string.

    Uses think=True so the model's chain-of-thought is routed to the separate
    message.thinking field; message.content contains only the clean final answer.

    Prints timing diagnostics: wall time, tokens generated, tokens/sec.
    """
    t0 = time.monotonic()

    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        think=True,   # route CoT to .thinking; content = clean answer
        options={
            # Total token budget (thinking + answer). qwen3:4b uses ~100–400 thinking
            # tokens for conversational queries; 1200 leaves ample room for the answer.
            "num_predict": config.OLLAMA_NUM_PREDICT,
        },
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
    )

    elapsed = time.monotonic() - t0

    # message.content is the clean answer (thinking was routed to .thinking field)
    content = _strip_think(response.message.content)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    try:
        thinking_field = getattr(response.message, "thinking", None)
        think_len      = len(thinking_field) if thinking_field else 0
        eval_tokens    = response.eval_count
        eval_secs      = response.eval_duration / 1e9
        prompt_tok     = response.prompt_eval_count
        tps            = eval_tokens / eval_secs if eval_secs > 0 else 0
        print(
            f"  🧠  [LLM] {elapsed:.1f}s wall  |  "
            f"prompt={prompt_tok}tok  gen={eval_tokens}tok  @ {tps:.0f}tok/s  "
            f"think={think_len}chars"
        )
    except Exception:
        print(f"  🧠  [LLM] {elapsed:.1f}s wall")

    return content
