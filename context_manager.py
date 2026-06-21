import json
import threading
import re
from datetime import datetime
from typing import Dict, Any

import ollama
import config

_state = {
    "current_topic": "",
    "entities": [],
    "last_user_message": "",
    "last_assistant_message": "",
    "updated_at": ""
}

_turn_count = 0

def get_state() -> Dict[str, Any]:
    global _state
    return _state

def get_context_prompt() -> str:
    global _state
    
    if not _state["current_topic"] and not _state["entities"]:
        return ""
        
    lines = []
    lines.append("\n\n── ACTIVE CONTEXT ─────────────────────────────────────────────────────────────")
    lines.append("Use this to resolve pronouns (it, they, he, she) or implicit references in the user's latest message.")
    
    if _state["current_topic"]:
        lines.append(f"Current Topic: {_state['current_topic']}")
        
    if _state["entities"]:
        for ent in _state["entities"]:
            t = ent.get("type", "entity")
            v = ent.get("value", "")
            lines.append(f"Recent Entity ({t}): {v}")
            
    return "\n".join(lines)

def clear_context():
    global _state, _turn_count
    _state = {
        "current_topic": "",
        "entities": [],
        "last_user_message": "",
        "last_assistant_message": "",
        "updated_at": ""
    }
    _turn_count = 0
    print("🧠 Context Cleared")

def update_context_async(user_msg: str, assistant_msg: str):
    global _turn_count
    _turn_count += 1
    
    if _turn_count > 20:
        clear_context()
        return
        
    thread = threading.Thread(target=_run_llm_extraction, args=(user_msg, assistant_msg))
    thread.daemon = True
    thread.start()

def _run_llm_extraction(user_msg: str, assistant_msg: str):
    global _state
    
    prompt = f"""
Extract the current topic and any concrete entities from the following exchange.
Output ONLY valid JSON. No markdown, no thinking, no explanations.

Current State:
Topic: {_state['current_topic']}
Entities: {json.dumps(_state['entities'])}

New Exchange:
User: {user_msg}
Assistant: {assistant_msg}

Required JSON format:
{{
  "current_topic": "short topic string (e.g. bike, travel, career, favorite colors)",
  "entities": [
    {{ "type": "entity_category", "value": "Entity Name" }}
  ]
}}
"""
    try:
        # Use qwen3 via Ollama
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        
        content = response["message"]["content"].strip()
        # Remove markdown blocks if model ignored instructions
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"```$", "", content).strip()
        
        # Handle reasoning model output
        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if match:
            content = match.group(1)
            
        data = json.loads(content)
        
        _state["current_topic"] = data.get("current_topic", _state["current_topic"])
        _state["entities"] = data.get("entities", [])
        _state["last_user_message"] = user_msg
        _state["last_assistant_message"] = assistant_msg
        _state["updated_at"] = datetime.now().isoformat()
        
        print("\n🧠 Context Updated")
        print(f"🧠 Topic: {_state['current_topic']}")
        for ent in _state["entities"]:
            print(f"🧠 Entity: {ent.get('value')}")
            
    except Exception as e:
        # Silently fail if JSON parse fails to not interrupt the user
        pass
