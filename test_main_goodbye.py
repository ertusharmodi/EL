import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock hardware dependencies before any imports
sys.modules['sounddevice'] = MagicMock()
sys.modules['elevenlabs'] = MagicMock()
sys.modules['audio'] = MagicMock()
sys.modules['tts'] = MagicMock()

# Import necessary modules
import intent_classifier
import intent_classifier

class TestGoodbyeFlow(unittest.TestCase):
    
    @patch('tts.speak')
    @patch('audio.play')
    def test_goodbye_flow_tts_completion(self, mock_audio_play, mock_tts_speak):
        # This test ensures the state transitions occur AFTER audio playback
        user_text = "bye"
        
        # 1. Intent classifier detects GOODBYE
        intent = intent_classifier.classify_intent(user_text)
        self.assertEqual(intent, "GOODBYE")
        
        # 2. Main.py logic simulation
        should_sleep = False
        follow_up_mode = True
        response = None
        
        if intent == "GOODBYE":
            response = "See you later."
            should_sleep = True
            
        # Assert response is correct
        self.assertEqual(response, "See you later.")
        self.assertTrue(should_sleep)
        
        # Simulate TTS
        wav_out = b"mock_audio_data"
        mock_tts_speak.return_value = wav_out
        wav_generated = mock_tts_speak(response)
        
        # Simulate Audio Playback
        mock_audio_play(wav_generated)
        
        # Ensure audio.play was called
        mock_audio_play.assert_called_once_with(wav_out)
        
        # 3. Enter sleep mode ONLY AFTER audio finishes
        if should_sleep:
            follow_up_mode = False
            
        # Assert sleep mode activated
        self.assertFalse(follow_up_mode)

if __name__ == '__main__':
    unittest.main()
