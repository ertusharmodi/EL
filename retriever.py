import re
from typing import Optional, Any
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
    
    # 1. Favorite <thing>
    match = re.search(r"what is my favo(?:u)?rite (.+)", clean_msg)
    if match:
        thing = match.group(1).strip()
        slug = re.sub(r"[^a-z0-9]+", "_", thing).strip("_")
        key = f"favorite_{slug}"
        
        val = memory_manager.get_value("preferences", key)
        if val:
            print("🧠 Memory Match Found")
            formatted = _format_value(val)
            is_plural = isinstance(val, list) and len(val) > 1
            if is_plural:
                ans = f"Your favorite {thing}s are {formatted}."
            else:
                ans = f"Your favorite {thing} is {formatted}."
            print("🧠 Memory Answer Generated")
            return ans
            
    # 2. Tell me about my preferences
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

    # 3. What do you know about me?
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
