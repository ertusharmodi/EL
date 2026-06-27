import unittest
from fast_path import route

class TestConversationalV2(unittest.TestCase):
    def test_multi_intent_messages(self):
        # 1. "How are you? What are you doing?" -> "I'm good. Just talking with you."
        resp, tag, sleep = route("How are you? What are you doing?")
        self.assertEqual(tag, "CONVO_MULTI")
        self.assertEqual(resp, "I'm good. Just talking with you.")
        self.assertFalse(sleep)

        # 2. "How are you and what are you doing?"
        resp, tag, sleep = route("How are you and what are you doing?")
        self.assertEqual(tag, "CONVO_MULTI")
        self.assertEqual(resp, "I'm good. Just talking with you.")
        self.assertFalse(sleep)

    def test_emotional_statements(self):
        # "I'm good but I miss you."
        resp, tag, sleep = route("I'm good but I miss you.")
        self.assertEqual(tag, "CONVO_MULTI")
        self.assertEqual(resp, "Glad to hear that. I've missed you too.")
        self.assertFalse(sleep)

        # "Thanks baby."
        resp, tag, sleep = route("Thanks baby.")
        self.assertEqual(tag, "CONVO_MULTI")
        self.assertEqual(resp, "You're welcome.")
        self.assertFalse(sleep)

        # "I'm also good."
        resp, tag, sleep = route("I'm also good.")
        self.assertEqual(tag, "CONVO_MULTI")
        self.assertEqual(resp, "Glad to hear that.")
        self.assertFalse(sleep)

    def test_conversational_combinations(self):
        # Mixed good/thanks/bye
        resp, tag, sleep = route("I'm good. Thanks! Bye.")
        self.assertEqual(tag, "CONVO_MULTI")
        self.assertEqual(resp, "Glad to hear that. You're welcome. See you later.")
        self.assertTrue(sleep)

    def test_fallback_on_extra_content(self):
        # "I'm good but turn off the lights." -> should NOT be CONVO_MULTI
        resp, tag, sleep = route("I'm good but turn off the lights.")
        # Might fall through to None or match something else if we had a fallback
        # Let's ensure tag is not CONVO_MULTI
        self.assertNotEqual(tag, "CONVO_MULTI")

if __name__ == "__main__":
    unittest.main()
