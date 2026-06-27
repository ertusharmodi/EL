# telemetry/emitter.py — no-op stub.
# Overlay has been removed. These functions are kept as stubs so that any
# future callers don't raise ImportError during the transition period.

def send_state(state: str, message: str = "") -> None:
    """No-op: overlay removed."""
    pass


def send_audio_level(level: float) -> None:
    """No-op: overlay removed."""
    pass
