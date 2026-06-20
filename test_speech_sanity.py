import unittest
from speech_sanity import check_sanity

class TestSpeechSanity(unittest.TestCase):
    
    def test_good_speech(self):
        # Normal duration, normal words, high confidence
        is_suspicious, _ = check_sanity("I am doing well today.", duration_sec=2.5, confidence=0.90)
        self.assertFalse(is_suspicious)
        
    def test_short_duration_hallucination(self):
        # 1.5 seconds but 10 words -> > 6.6 words/sec
        transcript = "I went to the store and bought a lot of apples."
        is_suspicious, reason = check_sanity(transcript, duration_sec=1.5, confidence=0.8)
        self.assertTrue(is_suspicious)
        self.assertIn("Unnatural speech rate", reason)
        
    def test_hardcoded_hallucination(self):
        # Even with good confidence and duration, hardcoded string fails it
        is_suspicious, reason = check_sanity("I'm also a bird.", duration_sec=3.0, confidence=0.9)
        self.assertTrue(is_suspicious)
        self.assertIn("known hallucination phrase", reason)

    def test_low_confidence_long_phrase(self):
        # 5 words, confidence 0.70
        is_suspicious, reason = check_sanity("This is a random sentence.", duration_sec=3.0, confidence=0.70)
        self.assertTrue(is_suspicious)
        self.assertIn("Low confidence", reason)
        
    def test_low_confidence_short_phrase(self):
        # 3 words, confidence 0.60 -> Should PASS because length <= 4
        is_suspicious, _ = check_sanity("Yes I am.", duration_sec=1.5, confidence=0.60)
        self.assertFalse(is_suspicious)

if __name__ == '__main__':
    unittest.main()
