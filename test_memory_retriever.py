import unittest
from unittest.mock import patch, MagicMock
import sys
sys.modules['ollama'] = MagicMock()

import memory_retriever

class TestMemoryRetriever(unittest.TestCase):
    
    def setUp(self):
        memory_retriever._cached_embeddings.clear()
        memory_retriever._cached_memory_state = ""
        
    def test_flatten_memory(self):
        mem = {
            "preferences": {"favorite_color": "blue"},
            "skills": {"tech_stack": ["React", "Python"]}
        }
        flat = memory_retriever._flatten_memory(mem)
        self.assertEqual(len(flat), 2)
        self.assertIn(("preferences", "favorite_color", "blue"), flat)
        self.assertIn(("skills", "tech_stack", ["React", "Python"]), flat)
        
    def test_cosine_similarity(self):
        # Identical vectors -> 1.0
        self.assertAlmostEqual(memory_retriever._cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        # Orthogonal vectors -> 0.0
        self.assertAlmostEqual(memory_retriever._cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        # Opposite vectors -> -1.0
        self.assertAlmostEqual(memory_retriever._cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)
        
    @patch('memory_retriever._get_embedding')
    def test_retrieve_relevant(self, mock_get_embedding):
        # We simulate vector embeddings where the query is very close to tech_stack,
        # somewhat close to favorite_color, and orthogonal to birthplace.
        def mock_embed(text):
            if "What technologies do I use?" in text:
                return [1.0, 0.0, 0.0]
            elif "tech_stack =" in text:
                return [0.9, 0.1, 0.0] # High similarity
            elif "favorite_color =" in text:
                return [0.5, 0.8, 0.0] # Moderate similarity (score = 0.5/sqrt(0.89) = ~0.53, still > 0.45)
            elif "birthplace =" in text:
                return [0.0, 0.0, 1.0] # Orthogonal (score = 0.0)
            return [0.0, 1.0, 0.0]
            
        mock_get_embedding.side_effect = mock_embed
        
        mem = {
            "skills": {"tech_stack": ["React", "Node.js"]},
            "preferences": {"favorite_color": "blue"},
            "personal": {"birthplace": "Hingona"}
        }
        
        # Request Top 1
        res = memory_retriever.retrieve_relevant("What technologies do I use?", mem, top_k=1)
        self.assertIn("skills", res)
        self.assertIn("tech_stack", res["skills"])
        self.assertNotIn("preferences", res)
        self.assertNotIn("personal", res)
        
        # Request Top 3
        # Should return tech_stack and favorite_color (since score > 0.45), but drop birthplace
        res2 = memory_retriever.retrieve_relevant("What technologies do I use?", mem, top_k=3)
        self.assertIn("skills", res2)
        self.assertIn("preferences", res2)
        self.assertNotIn("personal", res2)

if __name__ == '__main__':
    unittest.main()
