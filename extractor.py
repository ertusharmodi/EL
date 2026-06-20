import json
import ollama

import config
import memory

_EXTRACTOR_SYSTEM_PROMPT = """
You are a memory extraction assistant.
Extract any durable, long-term facts about the user from the following statement.
Ignore temporary states, feelings, or conversational filler.
Return a single JSON object where keys are the fact names (e.g., "name", "city", "favorite_color", "preferred_language") and values are simple string values.
If there are no durable facts, return an empty JSON object: {}
"""

def extract_and_save(user_text: str) -> None:
    """
    Run an asynchronous LLM pass to extract durable facts from the user's input,
    and save them directly to the profile memory.
    """
    try:
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            format="json",
            options={
                "temperature": 0.0,
                "num_predict": 200,
            },
            messages=[
                {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": f"User statement: {user_text}"}
            ]
        )
        
        # We asked for format="json", so it should parse safely.
        content = response.message.content.strip()
        if not content:
            return
            
        facts = json.loads(content)
        if facts and isinstance(facts, dict):
            memory.set_profile_fields(facts)
            
    except Exception as exc:
        print(f"  ⚠️   [MEMORY] Extraction failed: {exc}")
