import datetime
from typing import List, Dict, Optional
import memory

def get_current_timestamp() -> str:
    """Returns the current timezone-aware timestamp in ISO-8601 format."""
    return datetime.datetime.now().astimezone().isoformat()

def calculate_response_time_ms(start_time: float, end_time: float) -> int:
    """Calculates response time in milliseconds."""
    return int((end_time - start_time) * 1000)

def get_messages_for_date(date_str: str) -> List[Dict]:
    """Retrieves all messages for a specific ISO-8601 date string (e.g. '2026-06-21')."""
    history = memory.get_history()
    result = []
    for msg in history:
        ts = msg.get("timestamp")
        if ts and ts.startswith(date_str):
            result.append(msg)
    return result

def get_messages_between(start_iso: str, end_iso: str) -> List[Dict]:
    """Retrieves all messages between two ISO-8601 timestamps."""
    history = memory.get_history()
    result = []
    for msg in history:
        ts = msg.get("timestamp")
        if ts and start_iso <= ts <= end_iso:
            result.append(msg)
    return result

def get_recent_messages(limit: int) -> List[Dict]:
    """Retrieves the most recent N messages."""
    history = memory.get_history()
    return history[-limit:] if limit > 0 else []
