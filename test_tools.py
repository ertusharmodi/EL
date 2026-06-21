import unittest
from unittest.mock import patch

from tools import calculator_tool, datetime_tool, system_tool
import tool_router

class TestTools(unittest.TestCase):
    
    def test_calculator_tool(self):
        self.assertEqual(calculator_tool.execute("2+2"), "4")
        self.assertEqual(calculator_tool.execute("45*76"), "3420")
        self.assertEqual(calculator_tool.execute("calculate 900/12"), "75")
        self.assertEqual(calculator_tool.execute("what is 5 + 5?"), "10")
        
        # Non-math queries should return None
        self.assertIsNone(calculator_tool.execute("hello"))
        self.assertIsNone(calculator_tool.execute("what is your name"))
        # Just a number is not math
        self.assertIsNone(calculator_tool.execute("12"))
        
    def test_datetime_tool(self):
        # We can't strictly assert the exact string since time changes, but we check pattern
        time_resp = datetime_tool.execute("what time is it")
        self.assertIsNotNone(time_resp)
        self.assertIn("It's", time_resp)
        
        date_resp = datetime_tool.execute("today's date")
        self.assertIsNotNone(date_resp)
        self.assertIn("Today is", date_resp)
        
        # Non-datetime
        self.assertIsNone(datetime_tool.execute("hello"))
        
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_system_tool(self, mock_vm, mock_cpu):
        mock_cpu.return_value = 12.0
        mock_vm.return_value.percent = 45.0
        
        resp = system_tool.execute("memory usage")
        self.assertEqual(resp, "CPU is at 12.0% and RAM is at 45.0%.")
        
        resp_ram = system_tool.execute("ram usage")
        self.assertEqual(resp_ram, "CPU is at 12.0% and RAM is at 45.0%.")
        
        # Disk specifically
        with patch('psutil.disk_usage') as mock_disk:
            mock_disk.return_value.percent = 80.0
            resp_disk = system_tool.execute("disk usage")
            self.assertEqual(resp_disk, "CPU is at 12.0%, RAM is at 45.0%, and Disk is 80.0% full.")
            
        self.assertIsNone(system_tool.execute("hello"))
        
    def test_tool_router(self):
        # Route datetime
        t, r = tool_router.route_tool("what time is it")
        self.assertEqual(t, "datetime")
        self.assertIsNotNone(r)
        
        # Route math
        t, r = tool_router.route_tool("what is 10 * 10")
        self.assertEqual(t, "calculator")
        self.assertEqual(r, "100")
        
        # Route nothing
        t, r = tool_router.route_tool("hello eleven")
        self.assertIsNone(t)
        self.assertIsNone(r)

if __name__ == '__main__':
    unittest.main()
