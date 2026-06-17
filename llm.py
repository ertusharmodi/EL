# llm.py — Chat with qwen3 via Ollama, optimised for voice-assistant latency.
#
# qwen3 models are reasoning models that generate a chain-of-thought before answering.
# On Ollama 0.30.x, passing think=True routes the chain-of-thought into the separate
# message.thinking field; message.content contains ONLY the clean final answer.
#
# Measured latency (this machine, ~142 tok/s):
#   qwen3:1.7b — 1.1–1.9 s/turn  ← current model  (thinking=~500–900 chars)
#   qwen3:4b   — 32–46 s/turn                       (thinking=~8,000–12,000 chars)
#
# Switch OLLAMA_MODEL in config.py to change the model.

import os
import re
import time

import ollama

import config

# ── System prompt ─────────────────────────────────────────────────────────────
_PERSONALITY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "personality.txt",
)

try:
    with open(_PERSONALITY_FILE, "r", encoding="utf-8") as _f:
        _SYSTEM_PROMPT = _f.read().strip()
except FileNotFoundError:
    _SYSTEM_PROMPT = config.OLLAMA_SYSTEM_PROMPT

# ── Helpers ───────────────────────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean(text: str) -> str:
    """Strip any residual <think>...</think> blocks and whitespace."""
    return _THINK_RE.sub("", text).strip()


# ── Chat ──────────────────────────────────────────────────────────────────────

def chat(user_text: str) -> str:
    """
    Send user_text to the LLM and return the AI's response as a string.

    Uses think=True so the model's chain-of-thought is routed to message.thinking;
    message.content contains the clean final answer with no reasoning preamble.

    Falls back to stripping <think> tags from content if thinking field is empty.

    Prints timing diagnostics: wall time, tokens generated, tokens/sec,
    thinking field character count.
    """
    t0 = time.monotonic()

    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        think=True,    # route CoT to .thinking; content = clean answer
        options={
            # Token budget for thinking + answer combined.
            # qwen3:4b uses ~200–800 thinking tokens for conversational queries.
            # 5000 is a generous safety ceiling; the model stops naturally when done.
            "num_predict": config.OLLAMA_NUM_PREDICT,
        },
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
    )

    elapsed = time.monotonic() - t0

    # message.content is the clean answer when think=True works correctly.
    # _clean() strips residual <think> tags as a safety net.
    content = _clean(response.message.content)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    try:
        thinking_field = getattr(response.message, "thinking", None)
        think_chars    = len(thinking_field) if thinking_field else 0
        eval_tokens    = response.eval_count
        eval_secs      = response.eval_duration / 1e9
        prompt_tok     = response.prompt_eval_count
        tps            = eval_tokens / eval_secs if eval_secs > 0 else 0
        print(
            f"  🧠  [LLM] {elapsed:.1f}s  |  "
            f"prompt={prompt_tok}tok  gen={eval_tokens}tok @ {tps:.0f}tok/s  "
            f"thinking={think_chars}chars"
        )
    except Exception:
        print(f"  🧠  [LLM] {elapsed:.1f}s")

    return content
