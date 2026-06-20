import unittest
import retriever
import context_resolver

# We'll mock the memory module to isolate the tests from the actual disk state
import memory
import memory_manager

class TestMemoryRecall(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup mock memory state
        memory._profile = {
            "name": "Tushar"
        }
        memory_manager._memory_cache = {
            "preferences": {
                "favorite_color": ["Black", "White", "Blue"],
                "favorite_place": ["Pune", "Jaipur"]
            }
        }
        
    def test_single_value_memory(self):
        # 1. Ask for name
        ans = retriever.retrieve_direct_answer("What is my name?")
        self.assertEqual(ans, "Tushar.")
        
    def test_multi_value_memory(self):
        # 2. Ask for favorite place (singular)
        ans = retriever.retrieve_direct_answer("What is my favorite place?")
        self.assertEqual(ans, "Pune and Jaipur.")
        
        # Ask for favorite places (plural)
        ans2 = retriever.retrieve_direct_answer("What are my favorite places?")
        self.assertEqual(ans2, "Pune and Jaipur.")
        
        # Ask for favorite color (multiple items > 2)
        ans3 = retriever.retrieve_direct_answer("What's my favorite color?")
        self.assertEqual(ans3, "Black, White, and Blue.")

    def test_acknowledgement_statements(self):
        # 3. Test exact match acknowledgements
        acks = {"Got it.", "Noted.", "Exactly.", "Right.", "Makes sense."}
        
        ans = retriever.retrieve_direct_answer("Yes.")
        self.assertIn(ans, acks)
        
        ans2 = retriever.retrieve_direct_answer("Correct.")
        self.assertIn(ans2, acks)
        
        # Test regex match acknowledgements
        ans3 = retriever.retrieve_direct_answer("That's my favorite color.")
        self.assertIn(ans3, acks)

        ans4 = retriever.retrieve_direct_answer("Those are my favorite places.")
        self.assertIn(ans4, acks)

    def test_repeated_memory_questions(self):
        # 4. Context resolver handling "Tell me again"
        history = [
            {"role": "user", "content": "What is my favorite color?"},
            {"role": "assistant", "content": "Black, White, and Blue."}
        ]
        
        # Resolve the query
        resolved = context_resolver.resolve_context("Tell me again.", history)
        self.assertEqual(resolved, "What is my favorite color?")
        
        # Test resolving "And my favorite place?"
        resolved2 = context_resolver.resolve_context("And my favorite place?", history)
        self.assertEqual(resolved2, "What is my favorite place?")

        # Test resolving "And my name?"
        resolved3 = context_resolver.resolve_context("And my name?", history)
        self.assertEqual(resolved3, "What is my name?")
        

if __name__ == "__main__":
    unittest.main()
