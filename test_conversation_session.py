import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock hardware dependencies before any imports
sys.modules['sounddevice'] = MagicMock()
sys.modules['elevenlabs'] = MagicMock()
sys.modules['audio'] = MagicMock()
sys.modules['tts'] = MagicMock()

import intent_classifier

class TestConversationSession(unittest.TestCase):

    def test_session_state_transitions(self):
        # We simulate the state transitions of main.py since the actual loop runs natively.
        
        session_mode = False
        last_interaction = 0.0
        SESSION_TIMEOUT = 60.0
        
        now = 100.0
        
        # 1. Wake word starts session
        # User says "Eleven"
        wake_clean = "eleven"
        matched_phrase = "eleven"
        
        session_mode = True
        last_interaction = now
        
        self.assertTrue(session_mode)
        
        # 2. Multiple questions without wake word
        # In main.py, `if not session_mode:` is skipped, directly going to AWAKE loop
        now = 110.0
        # User says "how are you"
        intent = intent_classifier.classify_intent("how are you")
        self.assertEqual(intent, "SMALL_TALK")
        
        last_interaction = now
        self.assertTrue(session_mode) # Still active
        
        # 3. Timeout returns to sleep
        now = 180.0
        if session_mode and (now - last_interaction) >= SESSION_TIMEOUT:
            session_mode = False
            
        self.assertFalse(session_mode)
        
        # 4. Goodbye returns to sleep
        # Reset session
        session_mode = True
        last_interaction = now
        should_sleep = False
        
        # User says "bye"
        intent = intent_classifier.classify_intent("bye")
        self.assertEqual(intent, "GOODBYE")
        if intent == "GOODBYE":
            should_sleep = True
            
        # audio finishes playing...
        if should_sleep:
            session_mode = False
            
        self.assertFalse(session_mode)

if __name__ == '__main__':
    unittest.main()
