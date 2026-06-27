import unittest
from unittest.mock import patch, MagicMock

import intent_classifier
import response_policy
import tool_router

class TestPipeline(unittest.TestCase):
    
    @patch('llm.chat')
    @patch('extractor.run_extraction_and_save')
    @patch('subprocess.run')
    def test_tool_commands_bypass_llm(self, mock_run, mock_extractor, mock_llm_chat):
        """
        Prove that tool commands like 'Open Cursor' bypass the LLM completely.
        """
        
        commands = [
            "Open Cursor",
            "open chrome",
            "open whatsapp",
            "open vscode",
            "2+2",
            "what time is it"
        ]
        
        for user_text in commands:
            # Replicate main.py pipeline EXACTLY
            intent = intent_classifier.classify_intent(user_text)
            
            response = None
            should_sleep = False
            
            if intent in ("GOODBYE", "GREETING", "SMALL_TALK", "ACKNOWLEDGEMENT", "IDENTITY_QUERY", "THANKS"):
                response, should_sleep = response_policy.apply_policy(intent, user_text)
                
            # 2. Reminder Router
            if response is None:
                from reminders import manager as reminder_manager
                rem_intent, rem_response = reminder_manager.route_reminder(user_text)
                if rem_response:
                    response = rem_response
                    
            # 3. Tool Router
            if response is None:
                tool_name, tool_response = tool_router.route_tool(user_text)
                if tool_response:
                    response = tool_response
                    
            # 3. Memory Router
            if response is None and intent.startswith("MEMORY_"):
                response, should_sleep = response_policy.apply_policy(intent, user_text)
            
            # Background Fact Extraction
            if intent in ("MEMORY_REMEMBER", "MEMORY_UPDATE", "MEMORY_FORGET", "UNKNOWN"):
                mock_extractor(user_text, 1.0)
            
            # 4. LLM (Fallback)
            if response is not None:
                # Bypassed LLM!
                pass
            else:
                response = mock_llm_chat(user_text)
                
            # Assertions
            self.assertIsNotNone(response)
            mock_llm_chat.assert_not_called()
            
if __name__ == '__main__':
    unittest.main()
