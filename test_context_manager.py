import unittest
from unittest.mock import patch, MagicMock

import context_manager

class TestContextManager(unittest.TestCase):
    
    def setUp(self):
        context_manager.clear_context()
        
    def test_clear_context(self):
        context_manager._state["current_topic"] = "test"
        context_manager.clear_context()
        self.assertEqual(context_manager.get_state()["current_topic"], "")
        
    def test_get_context_prompt_empty(self):
        self.assertEqual(context_manager.get_context_prompt(), "")
        
    def test_get_context_prompt_with_data(self):
        context_manager._state["current_topic"] = "bike"
        context_manager._state["entities"] = [
            {"type": "motorcycle", "value": "Royal Enfield"}
        ]
        
        prompt = context_manager.get_context_prompt()
        self.assertIn("Current Topic: bike", prompt)
        self.assertIn("Recent Entity (motorcycle): Royal Enfield", prompt)
        
    @patch('context_manager.ollama.chat')
    def test_run_llm_extraction(self, mock_ollama_chat):
        # Mock the ollama response
        mock_response = {
            "message": {
                "content": '''```json
{
  "current_topic": "travel",
  "entities": [
    { "type": "location", "value": "Paris" }
  ]
}
```'''
            }
        }
        mock_ollama_chat.return_value = mock_response
        
        context_manager._run_llm_extraction("I want to visit Paris.", "That sounds lovely.")
        
        state = context_manager.get_state()
        self.assertEqual(state["current_topic"], "travel")
        self.assertEqual(len(state["entities"]), 1)
        self.assertEqual(state["entities"][0]["value"], "Paris")
        self.assertEqual(state["last_user_message"], "I want to visit Paris.")

    def test_turn_expiration(self):
        context_manager._turn_count = 20
        context_manager._state["current_topic"] = "test"
        
        # 21st turn should trigger clear
        context_manager.update_context_async("hello", "hi")
        
        self.assertEqual(context_manager.get_state()["current_topic"], "")
        self.assertEqual(context_manager._turn_count, 0)

if __name__ == '__main__':
    unittest.main()
