import unittest
import os
import json
import datetime
from unittest.mock import patch

import memory
import conversation
import config

class TestConversation(unittest.TestCase):
    def setUp(self):
        config.MEMORY_SHORT_TERM_FILE = "memory/test_short_term.json"
        if os.path.exists(config.MEMORY_SHORT_TERM_FILE):
            os.remove(config.MEMORY_SHORT_TERM_FILE)
        memory.clear()

    def tearDown(self):
        if os.path.exists(config.MEMORY_SHORT_TERM_FILE):
            os.remove(config.MEMORY_SHORT_TERM_FILE)
        memory.clear()

    def test_legacy_loading(self):
        # Create a legacy JSON without timestamps
        os.makedirs(os.path.dirname(config.MEMORY_SHORT_TERM_FILE), exist_ok=True)
        with open(config.MEMORY_SHORT_TERM_FILE, "w") as f:
            json.dump({
                "turns": [
                    {"role": "user", "content": "legacy user"},
                    {"role": "assistant", "content": "legacy assistant"}
                ]
            }, f)
        
        # Load and verify it doesn't crash and correctly loads 2 messages
        memory._load()
        history = memory.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["content"], "legacy user")
        self.assertNotIn("timestamp", history[0])

    def test_timestamp_injection(self):
        memory.add_turn("hello", "hi there", response_time_ms=420)
        
        history = memory.get_history()
        self.assertEqual(len(history), 2)
        
        user_msg = history[0]
        assistant_msg = history[1]
        
        self.assertEqual(user_msg["role"], "user")
        self.assertIn("timestamp", user_msg)
        
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertIn("timestamp", assistant_msg)
        self.assertEqual(assistant_msg["response_time_ms"], 420)
        
        # Verify ISO format and timezone
        user_dt = datetime.datetime.fromisoformat(user_msg["timestamp"])
        self.assertIsNotNone(user_dt.tzinfo)

    def test_conversation_queries(self):
        ts1 = "2026-06-21T10:00:00+00:00"
        ts2 = "2026-06-22T10:00:00+00:00"
        
        memory._history = [
            {"role": "user", "content": "a", "timestamp": ts1},
            {"role": "assistant", "content": "b", "timestamp": ts1},
            {"role": "user", "content": "c", "timestamp": ts2},
        ]
        
        # test date query
        res = conversation.get_messages_for_date("2026-06-21")
        self.assertEqual(len(res), 2)
        
        res2 = conversation.get_messages_for_date("2026-06-22")
        self.assertEqual(len(res2), 1)
        
        # test range query
        res3 = conversation.get_messages_between("2026-06-21T00", "2026-06-21T23")
        self.assertEqual(len(res3), 2)
        
        # test recent query
        res4 = conversation.get_recent_messages(1)
        self.assertEqual(len(res4), 1)
        self.assertEqual(res4[0]["content"], "c")

if __name__ == "__main__":
    unittest.main()
