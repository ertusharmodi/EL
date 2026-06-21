import unittest
from speech_corrector import correct_transcript

class TestSpeechCorrector(unittest.TestCase):
    
    def test_corrections(self):
        self.assertEqual(correct_transcript("bird"), "good")
        self.assertEqual(correct_transcript("by"), "bye")
        self.assertEqual(correct_transcript("buy"), "bye")
        self.assertEqual(correct_transcript("tusar"), "tushar")
        self.assertEqual(correct_transcript("larvel"), "laravel")
        self.assertEqual(correct_transcript("larval"), "laravel")
        self.assertEqual(correct_transcript("next yes"), "nextjs")
        self.assertEqual(correct_transcript("next jazz"), "nextjs")
        self.assertEqual(correct_transcript("node jay s"), "nodejs")
        self.assertEqual(correct_transcript("react nativee"), "react native")
        
    def test_case_insensitivity_and_preservation(self):
        # Should preserve original casing of non-matched words, but replace matched words
        self.assertEqual(correct_transcript("I am BIRD"), "I am good")
        self.assertEqual(correct_transcript("LarVal is awesome"), "laravel is awesome")
        
    def test_word_boundaries(self):
        # "by" shouldn't trigger inside "abyssal"
        self.assertEqual(correct_transcript("abyssal"), "abyssal")
        # "buy" shouldn't trigger inside "buyer"
        self.assertEqual(correct_transcript("buyer"), "buyer")

if __name__ == '__main__':
    unittest.main()
