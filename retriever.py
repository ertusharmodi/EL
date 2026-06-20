import re
from typing import Optional, Any
import memory
import memory_manager

def _format_value(value: Any) -> str:
    if isinstance(value, list):
        if len(value) == 1:
            return str(value[0])
        elif len(value) == 2:
            return f"{value[0]} and {value[1]}"
        else:
            return ", ".join(str(v) for v in value[:-1]) + f", and {value[-1]}"
    return str(value)

def retrieve_direct_answer(user_message: str) -> Optional[str]:
    print("🧠 Memory Query")
    
    clean_msg = user_message.strip().strip(".!? ").lower()

    # ── 1. Acknowledgements ────────────────────────────────────────────────
    ack_phrases = {"correct", "exactly", "yes", "yeah", "yep", "right", "makes sense"}
    if clean_msg in ack_phrases or re.match(r"^(?:that's|that is|those are) my ", clean_msg):
        import random
        acks = ["Got it.", "Noted.", "Exactly.", "Right.", "Makes sense."]
        print("🧠 Memory Match Found (Acknowledgement)")
        return random.choice(acks)

    # ── 2. Hardcoded conversational answers ────────────────────────────────
    hardcoded = {
        "what are you doing": "Just talking with you.",
        "okay": "Alright."
    }
    if clean_msg in hardcoded:
        print("🧠 Memory Match Found")
        print("🧠 Memory Answer Generated")
        return hardcoded[clean_msg]
        
    # ── 3. Exact Profile Matches (Name) ────────────────────────────────────
    if clean_msg in ("what's my name", "what is my name"):
        prof = memory.get_profile()
        name = prof.get("name")
        if name:
            print("🧠 Memory Match Found")
            print("🧠 Memory Answer Generated")
            # Only return the requested fact
            return f"{name}."

    # ── 4. Favorite <thing> ────────────────────────────────────────────────
    match = re.search(r"(?:what is|what's|what are) my favo(?:u)?rite (.+)", clean_msg)
    if match:
        thing = match.group(1).strip()
        slug = re.sub(r"[^a-z0-9]+", "_", thing).strip("_")
        key = f"favorite_{slug}"
        
        # Try exact key, then singular, then plural
        val = memory_manager.get_value("preferences", key)
        if not val and key.endswith("s"):
            val = memory_manager.get_value("preferences", key[:-1])
        if not val and not key.endswith("s"):
            val = memory_manager.get_value("preferences", key + "s")
            
        if val:
            print("🧠 Memory Match Found")
            formatted = _format_value(val)
            # Only return the requested fact, capitalized
            ans = formatted[0].upper() + formatted[1:] + "."
            print("🧠 Memory Answer Generated")
            return ans
            
    # ── 4. Tell me about my preferences ────────────────────────────────────
    if "preferences" in clean_msg and ("tell me" in clean_msg or "what are" in clean_msg or "what do you know" in clean_msg):
        mem = memory_manager.recall()
        prefs = mem.get("preferences", {})
        if prefs:
            print("🧠 Memory Match Found")
            items = []
            for k, v in prefs.items():
                thing = k.replace("favorite_", "").replace("_", " ")
                items.append(f"{thing} ({_format_value(v)})")
            ans = "Your preferences include: " + ", ".join(items) + "."
            print("🧠 Memory Answer Generated")
            return ans

    # ── 5. What do you know about me? ──────────────────────────────────────
    if clean_msg in ("what do you know about me", "tell me about me", "what do you know", "who am i"):
        mem = memory_manager.recall()
        facts = []
        for cat, items in mem.items():
            if isinstance(items, dict) and items:
                facts.append(cat)
                
        if facts:
            print("🧠 Memory Match Found")
            ans = f"I know several things about your {_format_value(facts)}. You can ask me about specific preferences, skills, or personal details."
            print("🧠 Memory Answer Generated")
            return ans

    return None

