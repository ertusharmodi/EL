import ast
import os
import glob
import unittest

class TestLoggingValidity(unittest.TestCase):
    def test_no_empty_logger_calls(self):
        """
        Scans the entire codebase for empty logger calls.
        Fails if any logger method (debug, info, warning, error, exception)
        is called without arguments.
        """
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        py_files = glob.glob(os.path.join(repo_dir, "**/*.py"), recursive=True)
        
        invalid_calls = []
        
        for filepath in py_files:
            # Skip site-packages or venv if they somehow get included
            if "site-packages" in filepath or "venv" in filepath or ".venv" in filepath:
                continue
                
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read(), filename=filepath)
                except SyntaxError:
                    continue
                    
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "logger":
                            method_name = node.func.attr
                            if method_name in ("debug", "info", "warning", "error", "exception"):
                                if not node.args and not node.keywords:
                                    invalid_calls.append(f"{os.path.basename(filepath)}:{node.lineno} -> logger.{method_name}()")
                                    
        if invalid_calls:
            calls_str = "\n".join(invalid_calls)
            self.fail(f"Found empty logger calls:\n{calls_str}")

if __name__ == '__main__':
    unittest.main()
