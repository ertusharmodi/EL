import re
from typing import List, Dict

def resolve_context(user_text: str, history: List[Dict[str, str]]) -> str:
    """
    Resolves conversational follow-ups into explicit standalone queries
    using fast heuristics and recent conversation history.
    """
    original_text = user_text.strip()
    clean_text = original_text.lower().strip(".!? ")

    resolved_text = original_text

    # 1. Handle "Repeat" intents (Tell me again, remind me, same)
    if clean_text in ("tell me again", "remind me", "same"):
        # Find the last user query to repeat it
        # history is a list of {"role": "user"|"assistant", "content": "..."}
        last_user_query = None
        # Get the last 10 messages (5 turns)
        recent_history = history[-10:] if len(history) > 10 else history
        for msg in reversed(recent_history):
            if msg.get("role") == "user":
                last_user_query = msg.get("content")
                break
        
        if last_user_query:
            resolved_text = last_user_query

    # 2. Handle "And my X?" / "What about my X?" intents
    else:
        # Regex to match follow-ups. e.g. "And my name?", "What about my favorite color?"
        match = re.match(r"^(?:and|what about|also)(?:\s+my)?\s+(.+)$", clean_text)
        if match:
            subject = match.group(1).strip()
            # Rewrite to an explicit question
            resolved_text = f"What is my {subject}?"

    # Print logs if a resolution occurred
    if resolved_text.lower() != original_text.lower():
        print("  🧠  Context Resolution")
        print(f"      Original: {original_text}")
        print(f"      Resolved: {resolved_text}")
        return resolved_text

    return original_text
