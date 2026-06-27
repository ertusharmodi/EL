# test_fast_path.py — Proves that deterministic queries never reach the LLM.
#
# route() returns (response, tag, should_sleep).
# All tests unpack the 3-tuple.

import sys
import time
import unittest
from unittest.mock import patch, MagicMock

# ── Stub heavy dependencies before importing project modules ──────────────────
sys.modules.setdefault("ollama", MagicMock())
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("numpy", MagicMock())
sys.modules.setdefault("faster_whisper", MagicMock())
sys.modules.setdefault("elevenlabs", MagicMock())


import fast_path
import memory
import memory_manager


# ── Helpers ───────────────────────────────────────────────────────────────────

def _route(query: str):
    """Convenience wrapper — always returns 3-tuple."""
    result = fast_path.route(query)
    assert len(result) == 3, f"route() must return 3 values, got {len(result)}"
    return result


# ── LLM bypass tests ──────────────────────────────────────────────────────────

class TestFastPathLLMBypass(unittest.TestCase):
    """Prove that fast-path queries never reach llm.chat."""

    def _assert_bypasses_llm(self, queries: list, mock_llm):
        for query in queries:
            with self.subTest(query=query):
                response, tag, _ = _route(query)
                self.assertIsNotNone(response, f"Expected a response for: {query!r}")
                mock_llm.assert_not_called()

    @patch("llm.chat")
    def test_identity_queries_bypass_llm(self, mock_llm):
        self._assert_bypasses_llm([
            "who are you",
            "what's your name",
            "what is your name",
            "what are you doing",
        ], mock_llm)

    @patch("llm.chat")
    def test_how_are_you_bypasses_llm(self, mock_llm):
        self._assert_bypasses_llm([
            "how are you",
            "how are you doing",
            "how about you",
        ], mock_llm)

    @patch("llm.chat")
    def test_greetings_bypass_llm(self, mock_llm):
        self._assert_bypasses_llm([
            "good morning",
            "good night",
            "hi",
            "hello",
        ], mock_llm)

    @patch("llm.chat")
    def test_thanks_bypass_llm(self, mock_llm):
        self._assert_bypasses_llm([
            "thanks",
            "thank you",
        ], mock_llm)

    @patch("llm.chat")
    def test_acknowledgements_bypass_llm(self, mock_llm):
        self._assert_bypasses_llm([
            "okay",
            "ok",
            "alright",
            "got it",
            "sounds good",
        ], mock_llm)

    @patch("llm.chat")
    def test_user_state_utterances_bypass_llm(self, mock_llm):
        self._assert_bypasses_llm([
            "i'm good",
            "i am good",
            "i'm fine",
            "i am fine",
            "i'm tired",
            "i'm happy",
        ], mock_llm)

    @patch("llm.chat")
    def test_goodbye_bypasses_llm(self, mock_llm):
        self._assert_bypasses_llm([
            "bye",
            "goodbye",
            "see you later",
        ], mock_llm)

    @patch("llm.chat")
    def test_time_query_bypasses_llm(self, mock_llm):
        self._assert_bypasses_llm([
            "what time is it",
            "what is the time",
        ], mock_llm)

    @patch("llm.chat")
    def test_date_query_bypasses_llm(self, mock_llm):
        self._assert_bypasses_llm([
            "what's today's date",
            "what is today's date",
        ], mock_llm)


# ── Correctness tests ─────────────────────────────────────────────────────────

class TestFastPathCorrectness(unittest.TestCase):

    # Identity
    def test_who_are_you(self):
        r, tag, sleep = _route("who are you")
        self.assertEqual(r, "I'm Eleven.")
        self.assertEqual(tag, "IDENTITY")
        self.assertFalse(sleep)

    def test_whats_your_name(self):
        r, tag, sleep = _route("what's your name")
        self.assertEqual(r, "Eleven.")
        self.assertFalse(sleep)

    def test_what_is_your_name(self):
        r, tag, _ = _route("what is your name")
        self.assertEqual(r, "Eleven.")

    def test_what_are_you_doing(self):
        r, tag, _ = _route("what are you doing")
        self.assertEqual(r, "Just talking with you.")
        self.assertEqual(tag, "IDENTITY")

    # Conversational
    def test_how_are_you(self):
        r, tag, sleep = _route("how are you")
        self.assertEqual(r, "I'm doing good. How about you?")
        self.assertEqual(tag, "CONVO")
        self.assertFalse(sleep)

    def test_how_about_you(self):
        r, tag, _ = _route("how about you")
        self.assertEqual(r, "I'm doing well.")
        self.assertEqual(tag, "CONVO")

    def test_im_good(self):
        r, tag, _ = _route("i'm good")
        self.assertEqual(r, "Good to hear!")

    def test_i_am_good(self):
        r, tag, _ = _route("i am good")
        self.assertEqual(r, "Good to hear!")

    def test_im_fine(self):
        r, tag, _ = _route("i'm fine")
        self.assertEqual(r, "Glad to hear it!")

    def test_im_tired(self):
        r, tag, _ = _route("i'm tired")
        self.assertEqual(r, "Get some rest.")

    def test_im_happy(self):
        r, tag, _ = _route("i'm happy")
        self.assertEqual(r, "That makes me happy too.")

    def test_good_morning(self):
        r, tag, sleep = _route("good morning")
        self.assertEqual(r, "Good morning!")
        self.assertFalse(sleep)

    def test_good_night(self):
        r, tag, _ = _route("good night")
        self.assertEqual(r, "Good night!")

    def test_thanks(self):
        r, tag, _ = _route("thanks")
        self.assertEqual(r, "You're welcome.")

    def test_thank_you(self):
        r, tag, _ = _route("thank you")
        self.assertEqual(r, "You're welcome.")

    def test_okay(self):
        r, tag, _ = _route("okay")
        self.assertEqual(r, "Alright.")

    def test_ok(self):
        r, tag, _ = _route("ok")
        self.assertEqual(r, "Alright.")

    # Goodbye — should_sleep=True
    def test_bye_triggers_sleep(self):
        r, tag, sleep = _route("bye")
        self.assertEqual(r, "See you later.")
        self.assertTrue(sleep)

    def test_goodbye_triggers_sleep(self):
        r, tag, sleep = _route("goodbye")
        self.assertTrue(sleep)

    def test_see_you_later_triggers_sleep(self):
        r, tag, sleep = _route("see you later")
        self.assertTrue(sleep)

    # Datetime
    def test_time_returns_non_empty(self):
        r, tag, _ = _route("what time is it")
        self.assertIsNotNone(r)
        self.assertIn("It's", r)
        self.assertEqual(tag, "DATETIME_TIME")

    def test_date_returns_non_empty(self):
        r, tag, _ = _route("what's today's date")
        self.assertIsNotNone(r)
        self.assertIn("Today is", r)
        self.assertEqual(tag, "DATETIME_DATE")

    # Memory
    def test_my_name_with_profile(self):
        original = memory._profile.copy()
        try:
            memory._profile["name"] = "Tushar"
            r, tag, _ = _route("what's my name")
            self.assertIsNotNone(r)
            self.assertIn("Tushar", r)
            self.assertEqual(tag, "MEMORY_NAME")
        finally:
            memory._profile.clear()
            memory._profile.update(original)

    def test_my_name_without_profile_returns_none(self):
        original = memory._profile.copy()
        try:
            memory._profile.pop("name", None)
            r, tag, sleep = _route("what's my name")
            self.assertIsNone(r)
            self.assertIsNone(tag)
            self.assertFalse(sleep)
        finally:
            memory._profile.clear()
            memory._profile.update(original)

    def test_fav_color_with_memory(self):
        mm = memory_manager._ensure_loaded()
        original_prefs = dict(mm.get("preferences", {}))
        try:
            mm.setdefault("preferences", {})["favorite_color"] = "blue"
            r, tag, _ = _route("what's my favorite color")
            self.assertIsNotNone(r)
            self.assertIn("Blue", r)
            self.assertEqual(tag, "MEMORY_FAV_COLOR")
        finally:
            mm.setdefault("preferences", {}).clear()
            mm["preferences"].update(original_prefs)

    def test_fav_color_without_memory_returns_none(self):
        mm = memory_manager._ensure_loaded()
        original_prefs = dict(mm.get("preferences", {}))
        try:
            mm.setdefault("preferences", {}).pop("favorite_color", None)
            mm["preferences"].pop("favorite_colors", None)
            r, tag, _ = _route("what's my favorite color")
            self.assertIsNone(r)
        finally:
            mm.setdefault("preferences", {}).clear()
            mm["preferences"].update(original_prefs)

    def test_fav_framework_with_memory(self):
        mm = memory_manager._ensure_loaded()
        original_prefs = dict(mm.get("preferences", {}))
        try:
            mm.setdefault("preferences", {})["favorite_framework"] = "Laravel"
            r, tag, _ = _route("what is my favorite framework")
            self.assertIsNotNone(r)
            self.assertIn("Laravel", r)
            self.assertEqual(tag, "MEMORY_FAV_FRAMEWORK")
        finally:
            mm.setdefault("preferences", {}).clear()
            mm["preferences"].update(original_prefs)

    def test_unknown_query_returns_none(self):
        r, tag, sleep = _route("tell me a joke")
        self.assertIsNone(r)
        self.assertIsNone(tag)
        self.assertFalse(sleep)


# ── Name-prefix stripping tests ───────────────────────────────────────────────

class TestFastPathPrefixStripping(unittest.TestCase):
    """Voiced queries often include the assistant's name as a prefix."""

    PREFIXED_CASES = [
        ("Eleven, how about you?",   "I'm doing well."),
        ("Eleven how about you",     "I'm doing well."),
        ("hey eleven, how are you",  "I'm doing good. How about you?"),
        ("Hi Eleven, good morning",  "Good morning!"),
        ("hello eleven, thanks",     "You're welcome."),
        ("Eleven, okay",             "Alright."),
        ("Eleven, bye",              "See you later."),
        ("11, how are you?",         "I'm doing good. How about you?"),
    ]

    def test_prefix_stripped_before_lookup(self):
        for raw, expected_response in self.PREFIXED_CASES:
            with self.subTest(query=raw):
                r, tag, _ = _route(raw)
                self.assertIsNotNone(r, f"No match for prefixed query: {raw!r}")
                self.assertEqual(r, expected_response, f"Wrong response for: {raw!r}")

    def test_eleven_bye_triggers_sleep(self):
        _, _, sleep = _route("Eleven, bye")
        self.assertTrue(sleep)


# ── Performance tests ─────────────────────────────────────────────────────────

class TestFastPathPerformance(unittest.TestCase):
    """All fast-path queries must resolve in < 300 ms (target < 500 ms)."""

    TARGET_MS = 300

    SIMPLE_QUERIES = [
        "who are you",
        "what's your name",
        "how are you",
        "how about you",
        "good morning",
        "good night",
        "thanks",
        "thank you",
        "okay",
        "bye",
        "see you later",
        "i am good",
        "i'm fine",
        "what time is it",
        "what's today's date",
        # Prefixed variants
        "Eleven, how about you?",
        "hey eleven, good morning",
    ]

    def test_response_time_under_300ms(self):
        for query in self.SIMPLE_QUERIES:
            with self.subTest(query=query):
                t0 = time.monotonic()
                r, _, _ = _route(query)
                elapsed_ms = (time.monotonic() - t0) * 1000
                self.assertIsNotNone(r, f"No response for {query!r}")
                self.assertLess(
                    elapsed_ms,
                    self.TARGET_MS,
                    f"Fast path took {elapsed_ms:.1f}ms for {query!r} — exceeds {self.TARGET_MS}ms target",
                )

    def test_memory_queries_under_300ms(self):
        original = memory._profile.copy()
        mm = memory_manager._ensure_loaded()
        original_prefs = dict(mm.get("preferences", {}))
        try:
            memory._profile["name"] = "Tushar"
            mm.setdefault("preferences", {})["favorite_color"] = "blue"
            mm["preferences"]["favorite_framework"] = "Laravel"
            for query in [
                "what's my name", "what is my name",
                "what's my favorite color", "what is my favorite color",
                "what is my favorite framework", "what's my favorite framework",
            ]:
                with self.subTest(query=query):
                    t0 = time.monotonic()
                    r, _, _ = _route(query)
                    elapsed_ms = (time.monotonic() - t0) * 1000
                    self.assertIsNotNone(r, f"No response for {query!r}")
                    self.assertLess(elapsed_ms, self.TARGET_MS)
        finally:
            memory._profile.clear()
            memory._profile.update(original)
            mm.setdefault("preferences", {}).clear()
            mm["preferences"].update(original_prefs)


# ── Normalisation tests ───────────────────────────────────────────────────────

class TestFastPathNormalization(unittest.TestCase):

    def test_trailing_question_mark_stripped(self):
        r, _, _ = _route("how are you?")
        self.assertIsNotNone(r)

    def test_case_insensitive(self):
        r, _, _ = _route("WHO ARE YOU")
        self.assertIsNotNone(r)

    def test_mixed_case(self):
        r, _, _ = _route("What's Your Name")
        self.assertIsNotNone(r)

    def test_trailing_period_stripped(self):
        r, _, _ = _route("thanks.")
        self.assertIsNotNone(r)

    def test_double_punctuation_stripped(self):
        r, _, _ = _route("okay!")
        self.assertIsNotNone(r)


if __name__ == "__main__":
    unittest.main()
