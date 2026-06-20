import unittest
from intent_classifier import classify_intent

class TestIntentClassifier(unittest.TestCase):
    
    def test_greeting(self):
        self.assertEqual(classify_intent("Hi"), "GREETING")
        self.assertEqual(classify_intent("hello!"), "GREETING")
        self.assertEqual(classify_intent("Good morning"), "GREETING")

    def test_thanks(self):
        self.assertEqual(classify_intent("Thanks."), "THANKS")
        self.assertEqual(classify_intent("thank you!"), "THANKS")
        self.assertEqual(classify_intent("Appreciate it"), "THANKS")

    def test_goodbye(self):
        self.assertEqual(classify_intent("Bye!"), "GOODBYE")
        self.assertEqual(classify_intent("goodbye."), "GOODBYE")
        self.assertEqual(classify_intent("See you"), "GOODBYE")

    def test_identity(self):
        self.assertEqual(classify_intent("Who are you?"), "IDENTITY")
        self.assertEqual(classify_intent("What's your name?"), "IDENTITY")
        
    def test_memory_query(self):
        self.assertEqual(classify_intent("What is my name?"), "MEMORY_QUERY")
        self.assertEqual(classify_intent("Do you remember?"), "MEMORY_QUERY")
        
    def test_memory_update(self):
        self.assertEqual(classify_intent("Remember this: I like apples"), "MEMORY_UPDATE")
        self.assertEqual(classify_intent("My favorite food is pizza"), "MEMORY_UPDATE")
        self.assertEqual(classify_intent("I was born in Texas"), "MEMORY_UPDATE")

    def test_small_talk(self):
        self.assertEqual(classify_intent("How are you?"), "SMALL_TALK")
        self.assertEqual(classify_intent("What are you doing?"), "SMALL_TALK")
        
    def test_acknowledgement(self):
        self.assertEqual(classify_intent("Okay."), "ACKNOWLEDGEMENT")
        self.assertEqual(classify_intent("I'll do it."), "ACKNOWLEDGEMENT")

    def test_unknown(self):
        self.assertEqual(classify_intent("What is the capital of France?"), "UNKNOWN")
        self.assertEqual(classify_intent("Turn on the lights"), "UNKNOWN")
        self.assertEqual(classify_intent("Let's talk about the weather"), "UNKNOWN")


if __name__ == '__main__':
    unittest.main()
