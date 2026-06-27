# logger.py — Centralised log-level gating for Eleven.
#
# Usage in any module:
#   import logger
#   logger.debug("per-chunk diagnostic")
#   logger.info("You: hello")
#   logger.warning("low confidence transcript")
#   logger.error("failed to load model")
#
# The active level is read from config.LOG_LEVEL at import time and can be
# changed at runtime by calling set_level().
#
# Level hierarchy (same as stdlib logging):
#   DEBUG=10  INFO=20  WARNING=30  ERROR=40
#
# When LOG_LEVEL=INFO, only INFO / WARNING / ERROR messages are printed.
# DEBUG messages are completely suppressed — zero string formatting cost
# because the check happens before the f-string is evaluated in the caller
# (callers should use `logger.debug(f"...")` only inside a guard or accept
# the minor f-string overhead; for hot loops prefer `if logger.is_debug():`).

import config

# ── Level constants ────────────────────────────────────────────────────────────

DEBUG   = 10
INFO    = 20
WARNING = 30
ERROR   = 40

_LEVEL_NAMES = {
    "DEBUG":   DEBUG,
    "INFO":    INFO,
    "WARNING": WARNING,
    "ERROR":   ERROR,
}

# ── Active level ───────────────────────────────────────────────────────────────

def _parse_level(value) -> int:
    if isinstance(value, int):
        return value
    name = str(value).upper().strip()
    if name not in _LEVEL_NAMES:
        raise ValueError(
            f"Unknown LOG_LEVEL {value!r}. Valid values: {list(_LEVEL_NAMES)}"
        )
    return _LEVEL_NAMES[name]


_active_level: int = _parse_level(getattr(config, "LOG_LEVEL", "INFO"))


def set_level(level) -> None:
    """Change the active log level at runtime."""
    global _active_level
    _active_level = _parse_level(level)


def get_level() -> int:
    return _active_level


# ── Convenience predicates ────────────────────────────────────────────────────

def is_debug() -> bool:
    return _active_level <= DEBUG

def is_info() -> bool:
    return _active_level <= INFO


# ── Emit helpers ──────────────────────────────────────────────────────────────

def debug(msg: str) -> None:
    if _active_level <= DEBUG:
        print(msg)

def info(msg: str) -> None:
    if _active_level <= INFO:
        print(msg)

def warning(msg: str) -> None:
    if _active_level <= WARNING:
        print(msg)

def error(msg: str) -> None:
    # Errors always print regardless of level.
    print(msg)
