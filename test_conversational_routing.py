import unittest
from unittest.mock import patch

import intent_classifier
import response_policy

class TestConversationalRouting(unittest.TestCase):
    """
    Consolidated test suite for the NLP pipeline:
    Intent Classification -> Response Policy deterministic mapping.
    """
    
    # ── 1. Intent Classification Priority Rules ──

    def test_intent_priorities(self):
        # 1. GOODBYE
        self.assertEqual(intent_classifier.classify_intent("bye"), "GOODBYE")
        self.assertEqual(intent_classifier.classify_intent("see you"), "GOODBYE")
        
        # 2. TIME_QUERY
        self.assertEqual(intent_classifier.classify_intent("what time is it"), "TIME_QUERY")
        self.assertEqual(intent_classifier.classify_intent("current time"), "TIME_QUERY")
        
        # 3. DATE_QUERY
        self.assertEqual(intent_classifier.classify_intent("today's date"), "DATE_QUERY")
        
        # 4. MEMORY_QUERY
        self.assertEqual(intent_classifier.classify_intent("what is my favorite framework?"), "MEMORY_QUERY")
        self.assertEqual(intent_classifier.classify_intent("what's my name"), "MEMORY_QUERY")
        self.assertEqual(intent_classifier.classify_intent("who is my best friend"), "MEMORY_QUERY")
        
        # 5. MEMORY_UPDATE
        self.assertEqual(intent_classifier.classify_intent("my favorite color is black"), "MEMORY_UPDATE")
        self.assertEqual(intent_classifier.classify_intent("remember this: i like apples"), "MEMORY_UPDATE")
        
        # 6. IDENTITY_QUERY
        self.assertEqual(intent_classifier.classify_intent("who are you"), "IDENTITY_QUERY")
        self.assertEqual(intent_classifier.classify_intent("what's your name"), "IDENTITY_QUERY")
        self.assertEqual(intent_classifier.classify_intent("what are you doing"), "IDENTITY_QUERY")
        
        # 7. SMALL_TALK / GREETING / THANKS / ACKNOWLEDGEMENT
        self.assertEqual(intent_classifier.classify_intent("hello!"), "GREETING")
        self.assertEqual(intent_classifier.classify_intent("thank you!"), "THANKS")
        self.assertEqual(intent_classifier.classify_intent("how are you?"), "SMALL_TALK")
        self.assertEqual(intent_classifier.classify_intent("okay."), "ACKNOWLEDGEMENT")

        # 8. LLM_FALLBACK (UNKNOWN)
        self.assertEqual(intent_classifier.classify_intent("what is the capital of france?"), "UNKNOWN")

    # ── 2. Response Policy Determinstic Mappings ──

    @patch('tools.datetime_tool.execute')
    def test_time_date_policy(self, mock_dt):
        mock_dt.return_value = "It's 10:00 AM."
        resp, _ = response_policy.apply_policy("TIME_QUERY", "what time is it")
        self.assertEqual(resp, "It's 10:00 AM.")
        mock_dt.assert_called_with("time")
        
        mock_dt.return_value = "Today is Monday."
        resp2, _ = response_policy.apply_policy("DATE_QUERY", "today's date")
        self.assertEqual(resp2, "Today is Monday.")
        mock_dt.assert_called_with("date")
        
    def test_identity_policy(self):
        resp, _ = response_policy.apply_policy("IDENTITY_QUERY", "who are you")
        self.assertEqual(resp, "I'm Eleven.")
        
        resp2, _ = response_policy.apply_policy("IDENTITY_QUERY", "what's your name")
        self.assertEqual(resp2, "Eleven.")
        
        resp3, _ = response_policy.apply_policy("IDENTITY_QUERY", "what are you doing")
        self.assertEqual(resp3, "Just talking with you.")
        
    @patch('retriever.retrieve_direct_answer')
    def test_memory_query_policy(self, mock_retriever):
        mock_retriever.return_value = "Laravel."
        resp, sleep = response_policy.apply_policy("MEMORY_QUERY", "what's my favorite framework?")
        self.assertEqual(resp, "Laravel.")
        self.assertFalse(sleep)

    def test_memory_update_policy(self):
        resp, sleep = response_policy.apply_policy("MEMORY_UPDATE", "I like apples")
        self.assertEqual(resp, "Got it.")
        self.assertFalse(sleep)
        
    def test_goodbye_policy(self):
        resp, sleep = response_policy.apply_policy("GOODBYE", "bye")
        self.assertEqual(resp, "See you later.")
        self.assertTrue(sleep) # Should trigger sleep mode
        
    def test_thanks_policy(self):
        resp, sleep = response_policy.apply_policy("THANKS", "thank you")
        self.assertEqual(resp, "You're welcome.")
        self.assertFalse(sleep)
        
    def test_small_talk_policy(self):
        resp, _ = response_policy.apply_policy("SMALL_TALK", "Nice")
        self.assertEqual(resp, "Nice.")
        
        resp2, _ = response_policy.apply_policy("SMALL_TALK", "i'm good")
        self.assertEqual(resp2, "Good to hear.")
        
        resp3, _ = response_policy.apply_policy("SMALL_TALK", "how are you?")
        self.assertEqual(resp3, "I'm good. How about you?")
        
    def test_acknowledgement_policy(self):
        resp, _ = response_policy.apply_policy("ACKNOWLEDGEMENT", "okay")
        self.assertEqual(resp, "Alright.")
        
        resp2, _ = response_policy.apply_policy("ACKNOWLEDGEMENT", "sounds good")
        self.assertEqual(resp2, "Sounds good.")
        
    def test_unknown_fallthrough(self):
        resp, sleep = response_policy.apply_policy("UNKNOWN", "Can you write me a poem?")
        self.assertIsNone(resp)
        self.assertFalse(sleep)

if __name__ == '__main__':
    unittest.main()
