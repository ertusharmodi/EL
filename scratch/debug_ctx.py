import sys
sys.path.append(".")
import context_manager
from unittest.mock import MagicMock
mock_ollama_chat = MagicMock()
mock_response = {
    "message": {
        "content": '''```json\n{"current_topic": "travel", "entities": [{"type": "location", "value": "Paris"}]}\n```'''
    }
}
mock_ollama_chat.return_value = mock_response
context_manager.ollama.chat = mock_ollama_chat
try:
    context_manager._run_llm_extraction("I want to visit Paris.", "That sounds lovely.")
    print(context_manager.get_state())
except Exception as e:
    print("ERROR:", e)
