import unittest
import os
import json
from unittest.mock import patch

from reminders import storage, manager

class TestReminders(unittest.TestCase):
    def setUp(self):
        # Use a dummy JSON file for testing
        storage.REMINDERS_FILE = "reminders/test_reminders.json"
        storage._reminders_cache = None
        if os.path.exists(storage.REMINDERS_FILE):
            os.remove(storage.REMINDERS_FILE)

    def tearDown(self):
        storage._reminders_cache = None
        if os.path.exists(storage.REMINDERS_FILE):
            os.remove(storage.REMINDERS_FILE)

    def test_create_reminder(self):
        # Direct parsing logic test
        t, task = manager.parse_time_and_task("remind me in 2 hours to push my code")
        self.assertEqual(t, "in 2 hours")
        self.assertEqual(task, "push my code")
        
        # Test full creation flow
        intent, resp = manager.route_reminder("remind me tomorrow at 9 AM to go to the gym")
        self.assertEqual(intent, "REMINDER_CREATE")
        self.assertEqual(resp, "Reminder created.")
        
        rems = storage.list_reminders()
        self.assertEqual(len(rems), 1)
        self.assertEqual(rems[0]["title"], "go to the gym")
        self.assertFalse(rems[0]["completed"])

    def test_list_reminders(self):
        # No reminders
        intent, resp = manager.route_reminder("what are my reminders")
        self.assertEqual(intent, "REMINDER_LIST")
        self.assertEqual(resp, "You have no reminders.")
        
        # Create 2 reminders
        storage.create_reminder("task 1", "2026-06-22T10:00:00")
        storage.create_reminder("task 2", "2026-06-22T11:00:00")
        
        intent, resp = manager.route_reminder("show reminders")
        self.assertEqual(intent, "REMINDER_LIST")
        self.assertEqual(resp, "You have 2 reminders.")

    def test_complete_reminder(self):
        storage.create_reminder("gym", "2026-06-22T10:00:00")
        storage.create_reminder("code", "2026-06-22T11:00:00")
        
        # Complete by index
        intent, resp = manager.route_reminder("mark reminder 1 complete")
        self.assertEqual(intent, "REMINDER_COMPLETE")
        self.assertEqual(resp, "Reminder marked complete.")
        
        # Now there is 1 active left
        self.assertEqual(len(storage.list_reminders()), 1)
        
        # Complete by title
        intent, resp = manager.route_reminder("complete my code reminder")
        self.assertEqual(intent, "REMINDER_COMPLETE")
        self.assertEqual(resp, "Reminder marked complete.")
        
        self.assertEqual(len(storage.list_reminders()), 0)

    def test_delete_reminder(self):
        storage.create_reminder("gym", "2026-06-22T10:00:00")
        
        intent, resp = manager.route_reminder("delete reminder 1")
        self.assertEqual(intent, "REMINDER_DELETE")
        self.assertEqual(resp, "Reminder deleted.")
        
        self.assertEqual(len(storage.list_reminders()), 0)

    def test_persistence(self):
        storage.create_reminder("persistent task", "2026-06-22T10:00:00")
        
        # Load directly from storage like a restart
        rems = storage.load_reminders()
        self.assertEqual(len(rems), 1)
        self.assertEqual(rems[0]["title"], "persistent task")
        
    @patch('reminders.scheduler.tts.speak')
    @patch('reminders.scheduler.audio.play')
    def test_scheduler(self, mock_play, mock_speak):
        from reminders import scheduler
        import time
        from datetime import datetime, timedelta
        
        # Create a reminder due 1 second ago to force immediate execution
        now = datetime.now()
        due = (now - timedelta(seconds=1)).isoformat()
        storage.create_reminder("Drink water", due)
        
        # We'll run one iteration of the poll loop synchronously instead of spawning a thread
        # just to prove the logic without arbitrary sleep waits.
        # But wait, the loop is infinite. We can't call _poll_loop() directly.
        # Let's extract the inside of the loop into a separate function, or we can just mock time.sleep
        # to raise an Exception to break the loop, or better, we can just write the inner logic test.
        # Actually, let's just patch time.sleep to raise a StopIteration so it runs exactly once.
        with patch('time.sleep', side_effect=StopIteration):
            try:
                scheduler._poll_loop()
            except StopIteration:
                pass
                
        # Verify TTS was called
        mock_speak.assert_called_with("Time to Drink water.")
        self.assertTrue(mock_play.called)
        
        # Verify it was marked completed
        active = storage.list_reminders()
        self.assertEqual(len(active), 0)
        
        # If we run it again, it shouldn't fire again
        mock_speak.reset_mock()
        with patch('time.sleep', side_effect=StopIteration):
            try:
                scheduler._poll_loop()
            except StopIteration:
                pass
                
        mock_speak.assert_not_called()

    @patch('memory.get_profile')
    def test_generate_reminder_speech(self, mock_profile):
        from reminders import scheduler
        
        # Default mock: no name
        mock_profile.return_value = {}
        
        # Verb tasks
        self.assertEqual(scheduler.generate_reminder_speech("drink water"), "Time to drink water.")
        self.assertEqual(scheduler.generate_reminder_speech("call dad"), "Time to call dad.")
        self.assertEqual(scheduler.generate_reminder_speech("push code"), "Time to push code.")
        self.assertEqual(scheduler.generate_reminder_speech("renew your passport"), "Time to renew your passport.")
        
        # Non-verb tasks
        self.assertEqual(scheduler.generate_reminder_speech("doctor appointment"), "Just a reminder: doctor appointment.")
        self.assertEqual(scheduler.generate_reminder_speech("the oven is on"), "Just a reminder: the oven is on.")
        
        # Name injection for important keywords
        mock_profile.return_value = {"name": "Tushar"}
        
        # Non-verb but important
        self.assertEqual(
            scheduler.generate_reminder_speech("doctor appointment"), 
            "Tushar, just a reminder: doctor appointment."
        )
        
        # Verb and important
        self.assertEqual(
            scheduler.generate_reminder_speech("call mom"), 
            "Tushar, time to call mom."
        )
        
        # Important but no name in profile
        mock_profile.return_value = {}
        self.assertEqual(
            scheduler.generate_reminder_speech("call mom"), 
            "Time to call mom."
        )

    def test_history(self):
        from reminders import manager
        from datetime import datetime
        
        storage.HISTORY_FILE = "reminders/test_history.json"
        if os.path.exists(storage.HISTORY_FILE):
            os.remove(storage.HISTORY_FILE)
            
        now = datetime.now().isoformat()
        
        # Test adding to history
        reminder1 = {"id": "1", "title": "Drink water", "created_at": now, "due_at": now}
        storage.add_to_history(reminder1, "completed")
        
        # Add another with explicit fired_at slightly later to test sorting
        reminder2 = {"id": "2", "title": "Call mom", "created_at": now, "due_at": now}
        storage.add_to_history(reminder2, "completed")
        
        history = storage.get_history()
        self.assertEqual(len(history), 2)
        # Verify descending sort (Call mom should be first because it was added later)
        self.assertEqual(history[0]["title"], "Call mom")
        
        # Test: What reminders fired today?
        intent, resp = manager.route_reminder("what reminders fired today")
        self.assertEqual(intent, "HISTORY_QUERY")
        self.assertIn("Drink water", resp)
        self.assertIn("Call mom", resp)
        
        # Test: How many reminders fired today?
        intent, resp = manager.route_reminder("how many reminders fired today")
        self.assertEqual(intent, "HISTORY_QUERY")
        self.assertEqual(resp, "2 reminders fired today.")
        
        # Test: When did you remind me to
        intent, resp = manager.route_reminder("when did you remind me to drink water")
        self.assertEqual(intent, "HISTORY_QUERY")
        self.assertIn("I reminded you to Drink water at", resp)
        
        # Test: Show reminder history
        intent, resp = manager.route_reminder("show reminder history")
        self.assertEqual(intent, "HISTORY_QUERY")
        self.assertIn("Call mom", resp)
        
        if os.path.exists(storage.HISTORY_FILE):
            os.remove(storage.HISTORY_FILE)

if __name__ == '__main__':
    unittest.main()
