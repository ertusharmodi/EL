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

        
        # 4. MEMORY_QUERY
        self.assertEqual(intent_classifier.classify_intent("what is my favorite framework?"), "MEMORY_QUERY")
        self.assertEqual(intent_classifier.classify_intent("what's my name"), "MEMORY_QUERY")
        self.assertEqual(intent_classifier.classify_intent("who is my best friend"), "MEMORY_QUERY")
        
        # 5. MEMORY_REMEMBER
        self.assertEqual(intent_classifier.classify_intent("remember that my favorite bike is Royal Enfield Himalayan"), "MEMORY_REMEMBER")
        self.assertEqual(intent_classifier.classify_intent("remember this: i like apples"), "MEMORY_REMEMBER")
        
        # 5b. MEMORY_UPDATE
        self.assertEqual(intent_classifier.classify_intent("update my favorite color to red"), "MEMORY_UPDATE")
        self.assertEqual(intent_classifier.classify_intent("change my name to john"), "MEMORY_UPDATE")
        
        # 5c. MEMORY_FORGET
        self.assertEqual(intent_classifier.classify_intent("forget my favorite color"), "MEMORY_FORGET")
        self.assertEqual(intent_classifier.classify_intent("delete my history"), "MEMORY_FORGET")
        
        # 5d. MEMORY_SUMMARY
        self.assertEqual(intent_classifier.classify_intent("what do you know about me"), "MEMORY_SUMMARY")
        self.assertEqual(intent_classifier.classify_intent("list my preferences"), "MEMORY_SUMMARY")
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

    def test_memory_remember_policy(self):
        resp, sleep = response_policy.apply_policy("MEMORY_REMEMBER", "I like apples")
        self.assertEqual(resp, "Got it.")
        self.assertFalse(sleep)
        
    def test_memory_update_policy(self):
        resp, sleep = response_policy.apply_policy("MEMORY_UPDATE", "change my favorite color to red")
        self.assertEqual(resp, "Updated.")
        self.assertFalse(sleep)
        
    def test_memory_forget_policy(self):
        resp, sleep = response_policy.apply_policy("MEMORY_FORGET", "forget my favorite color")
        self.assertEqual(resp, "Forgotten.")
        self.assertFalse(sleep)
        
    @patch('memory_manager.get_summary')
    def test_memory_summary_policy(self, mock_summary):
        mock_summary.return_value = "Name: John"
        resp, sleep = response_policy.apply_policy("MEMORY_SUMMARY", "what do you know about me")
        self.assertEqual(resp, "Name: John")
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
