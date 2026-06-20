"""
Unit tests for the Eleven hybrid memory extraction system.

Run: python -m unittest test_extractor -v
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import config
import extractor
import llm_extractor
import memory_manager
import regex_extractor


def _sorted_memories(memories):
    return sorted(
        (
            m["category"],
            m["key"],
            tuple(m["value"]) if isinstance(m["value"], list) else m["value"],
        )
        for m in memories
    )


class TestRegexExtraction(unittest.TestCase):
    """Regex stage — no disk I/O, no LLM."""

    def test_favorite_color(self):
        result = regex_extractor.extract_memories_regex("My favorite color is black")
        self.assertEqual(
            _sorted_memories(result),
            [("preferences", "favorite_color", "black")],
        )

    def test_favourite_british_spelling(self):
        result = regex_extractor.extract_memories_regex("My favourite color is black")
        self.assertEqual(result[0]["key"], "favorite_color")
        self.assertEqual(result[0]["value"], "black")

    def test_favorite_framework(self):
        result = regex_extractor.extract_memories_regex("My favorite framework is Laravel")
        self.assertEqual(result[0]["key"], "favorite_framework")

    def test_birthplace_explicit(self):
        result = regex_extractor.extract_memories_regex("My birthplace is Harmangrad Rajasthan")
        self.assertEqual(
            _sorted_memories(result),
            [("personal", "birthplace", "Harmangrad Rajasthan")],
        )

    def test_birthplace_was_born_in(self):
        result = regex_extractor.extract_memories_regex("I was born in Hingona Rajasthan.")
        self.assertEqual(result[0]["category"], "personal")
        self.assertEqual(result[0]["key"], "birthplace")

    def test_birthplace_im_from(self):
        result = regex_extractor.extract_memories_regex("I'm from London")
        self.assertEqual(result[0]["value"], "London")

    def test_profession(self):
        result = regex_extractor.extract_memories_regex("I am a software engineer")
        self.assertEqual(result[0]["value"], "Software Engineer")

    def test_skills_i_use(self):
        result = regex_extractor.extract_memories_regex("I use Laravel, PHP, Node.js and React")
        self.assertEqual(result[0]["key"], "tech_stack")
        self.assertEqual(len(result[0]["value"]), 4)

    def test_skills_i_love(self):
        result = regex_extractor.extract_memories_regex("I love Laravel and Node.js.")
        self.assertEqual(sorted(result[0]["value"]), sorted(["Laravel", "Node.js"]))

    def test_goal(self):
        result = regex_extractor.extract_memories_regex("My goal is to build a voice assistant")
        self.assertEqual(result[0]["category"], "goals")

    def test_project(self):
        result = regex_extractor.extract_memories_regex("My main project is Eleven")
        self.assertEqual(result[0]["category"], "projects")

    def test_multi_profession_birthplace_skills(self):
        msg = "I am a software engineer from Jaipur and I use Laravel and Node.js"
        result = regex_extractor.extract_memories_regex(msg)
        keys = {(m["category"], m["key"]) for m in result}
        self.assertIn(("personal", "profession"), keys)
        self.assertIn(("personal", "birthplace"), keys)
        self.assertIn(("skills", "tech_stack"), keys)

    def test_no_match_hungry(self):
        self.assertEqual(regex_extractor.extract_memories_regex("I am hungry"), [])

    def test_regex_confidence_is_full(self):
        result = regex_extractor.extract_memories_regex("My favorite color is black")
        self.assertEqual(result[0]["confidence"], 1.0)
        self.assertEqual(result[0]["source"], "regex")


class TestLLMExtraction(unittest.TestCase):
    """LLM stage — mocked, no Ollama required."""

    def test_parse_valid_json(self):
        raw = '{"memories":[{"category":"goals","key":"business_goal","value":"build a SaaS company","confidence":0.92}]}'
        parsed = llm_extractor._parse_json_response(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(len(parsed["memories"]), 1)

    def test_parse_json_with_fences(self):
        raw = '```json\n{"memories":[]}\n```'
        parsed = llm_extractor._parse_json_response(raw)
        self.assertEqual(parsed, {"memories": []})

    def test_normalize_memory_item(self):
        item = {
            "category": "relationships",
            "key": "sister",
            "value": "teacher",
            "confidence": 0.95,
        }
        result = llm_extractor._normalize_memory_item(item)
        self.assertEqual(result["category"], "relationships")
        self.assertEqual(result["source"], "llm")

    def test_rejects_low_confidence_format(self):
        item = {"category": "personal", "key": "name", "value": "Bob", "confidence": "high"}
        self.assertIsNone(llm_extractor._normalize_memory_item(item))

    @patch("llm_extractor.ollama.chat")
    def test_extract_business_goal(self, mock_chat):
        mock_chat.return_value.message.content = json.dumps(
            {
                "memories": [
                    {
                        "category": "goals",
                        "key": "business_goal",
                        "value": "build a SaaS company",
                        "confidence": 0.92,
                    }
                ]
            }
        )
        result = llm_extractor.extract_memories_llm(
            "I want to build a SaaS company in the future."
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["key"], "business_goal")
        self.assertEqual(result[0]["confidence"], 0.92)

    @patch("llm_extractor.ollama.chat")
    def test_extract_relationship_nested(self, mock_chat):
        mock_chat.return_value.message.content = json.dumps(
            {
                "memories": [
                    {
                        "category": "relationships",
                        "key": "friend_rahul.location",
                        "value": "Jaipur",
                        "confidence": 0.93,
                    }
                ]
            }
        )
        result = llm_extractor.extract_memories_llm("My friend Rahul lives in Jaipur.")
        self.assertEqual(result[0]["key"], "friend_rahul.location")

    @patch("llm_extractor.ollama.chat")
    def test_extract_empty_for_casual(self, mock_chat):
        mock_chat.return_value.message.content = '{"memories":[]}'
        result = llm_extractor.extract_memories_llm("What's the weather like?")
        self.assertEqual(result, [])


class TestHybridMerge(unittest.TestCase):
    """Regex + LLM merge logic."""

    @patch("llm_extractor.extract_memories_llm")
    def test_regex_wins_on_duplicate_key(self, mock_llm):
        # We test a non-list-like key to ensure exact match preferences regex
        mock_llm.return_value = [
            {
                "category": "personal",
                "key": "birthplace",
                "value": "Harmangrad",  # shorter, would normally be overwritten if LLM was "Harmangrad Rajasthan"
                "confidence": 0.9,
                "source": "llm",
            }
        ]
        # Regex extracts "Harmangrad Rajasthan"
        result = extractor.extract_memories("My birthplace is Harmangrad Rajasthan")
        place = next(m for m in result if m["key"] == "birthplace")
        self.assertEqual(place["value"], "Harmangrad Rajasthan")
        self.assertEqual(place["source"], "regex")

    @patch("llm_extractor.extract_memories_llm")
    def test_llm_fills_gaps(self, mock_llm):
        mock_llm.return_value = [
            {
                "category": "preferences",
                "key": "work_style",
                "value": "remote",
                "confidence": 0.9,
                "source": "llm",
            }
        ]
        result = extractor.extract_memories("I prefer remote work.")
        keys = {m["key"] for m in result}
        self.assertIn("work_style", keys)


class TestRunExtractionAndSave(unittest.TestCase):
    """Persistence, STT gating, confidence gating, deduplication, updates."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._memory_file = os.path.join(self._tmpdir.name, "long_term.json")
        self._patcher = patch.object(config, "MEMORY_LONG_TERM_FILE", self._memory_file)
        self._patcher.start()
        memory_manager._memory_cache = None
        memory_manager.load_memory()

    def tearDown(self):
        self._patcher.stop()
        memory_manager._memory_cache = None
        self._tmpdir.cleanup()

    @patch("llm_extractor.extract_memories_llm", return_value=[])
    def test_saves_new_favorite(self, _mock_llm):
        buf = io.StringIO()
        with redirect_stdout(buf):
            extractor.run_extraction_and_save("My favorite color is black", stt_confidence=0.95)
        self.assertEqual(
            memory_manager.get_value("preferences", "favorite_color"),
            "black",
        )
        output = buf.getvalue()
        self.assertIn("Extracted Memory", output)
        self.assertIn("Saved Memory", output)
        self.assertIn("Confidence", output)

    @patch("llm_extractor.extract_memories_llm", return_value=[])
    def test_skips_duplicate_favorite(self, _mock_llm):
        extractor.run_extraction_and_save("My favorite color is black", stt_confidence=1.0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            extractor.run_extraction_and_save("My favorite color is black", stt_confidence=1.0)
        output = buf.getvalue()
        self.assertIn("Extracted Memory", output)
        self.assertNotIn("Saved Memory", output)
        self.assertNotIn("Updated Memory", output)

    @patch("llm_extractor.extract_memories_llm", return_value=[])
    def test_updates_favorite_on_change(self, _mock_llm):
        extractor.run_extraction_and_save("My favorite color is black", stt_confidence=1.0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            extractor.run_extraction_and_save("My favorite color is blue", stt_confidence=1.0)
        self.assertEqual(
            memory_manager.get_value("preferences", "favorite_color"),
            ["black", "blue"],
        )
        self.assertIn("Updated Memory", buf.getvalue())

    @patch("llm_extractor.extract_memories_llm", return_value=[])
    def test_merges_tech_stack(self, _mock_llm):
        extractor.run_extraction_and_save("I use Laravel, PHP and React", stt_confidence=1.0)
        extractor.run_extraction_and_save("I mostly work with Node.js and Laravel", stt_confidence=1.0)
        stack = memory_manager.get_value("skills", "tech_stack")
        self.assertEqual(
            sorted(stack),
            sorted(["Laravel", "PHP", "React", "Node.js"]),
        )

    @patch("llm_extractor.extract_memories_llm", return_value=[])
    def test_low_stt_confidence_skips_all(self, _mock_llm):
        buf = io.StringIO()
        with redirect_stdout(buf):
            extractor.run_extraction_and_save("My favorite color is black", stt_confidence=0.75)
        self.assertIn("Memory save skipped due to low STT confidence", buf.getvalue())
        self.assertIsNone(memory_manager.get_value("preferences", "favorite_color"))

    @patch("llm_extractor.extract_memories_llm")
    def test_low_llm_confidence_skips_save(self, mock_llm):
        mock_llm.return_value = [
            {
                "category": "goals",
                "key": "business_goal",
                "value": "build a SaaS company",
                "confidence": 0.65,
                "source": "llm",
            }
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            extractor.run_extraction_and_save(
                "I want to build a SaaS company in the future.",
                stt_confidence=1.0,
            )
        output = buf.getvalue()
        self.assertIn("Extracted Memory", output)
        self.assertIn("Skipped", output)
        self.assertIsNone(memory_manager.get_value("goals", "business_goal"))

    @patch("llm_extractor.extract_memories_llm")
    def test_high_llm_confidence_saves(self, mock_llm):
        mock_llm.return_value = [
            {
                "category": "relationships",
                "key": "sister",
                "value": "teacher",
                "confidence": 0.95,
                "source": "llm",
            }
        ]
        extractor.run_extraction_and_save("My sister is a teacher.", stt_confidence=1.0)
        self.assertEqual(memory_manager.get_value("relationships", "sister"), "teacher")

    @patch("llm_extractor.extract_memories_llm")
    def test_nested_relationship_saved(self, mock_llm):
        mock_llm.return_value = [
            {
                "category": "relationships",
                "key": "friend_rahul.location",
                "value": "Jaipur",
                "confidence": 0.93,
                "source": "llm",
            }
        ]
        extractor.run_extraction_and_save("My friend Rahul lives in Jaipur.", stt_confidence=1.0)
        stored = memory_manager.recall()["relationships"]
        self.assertEqual(stored["friend_rahul"]["location"], "Jaipur")

    @patch("llm_extractor.extract_memories_llm", return_value=[])
    def test_birthplace_personal_category(self, _mock_llm):
        extractor.run_extraction_and_save(
            "My birthplace is Harmangrad Rajasthan",
            stt_confidence=1.0,
        )
        self.assertEqual(
            memory_manager.get_value("personal", "birthplace"),
            "Harmangrad Rajasthan",
        )


if __name__ == "__main__":
    unittest.main()
