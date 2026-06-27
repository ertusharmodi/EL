import re

with open("main.py", "r") as f:
    code = f.read()

code = code.replace("import fast_path\n", "import fast_path\nimport logger\n")

# Replace all print( with logger.debug(
code = code.replace("print(", "logger.debug(")

# Now fix the ones that should be info or warning
code = code.replace(
    'logger.debug("\\n🤖  Eleven is ready',
    'logger.info("\\n🤖  Eleven is ready'
)

code = code.replace(
    'logger.debug(f"  You  : {user_text} -> {resolved_user_text}")',
    'logger.info(f"  You  : {user_text} -> {resolved_user_text}")'
)

code = code.replace(
    'logger.debug(f"  You  : {user_text}")',
    'logger.info(f"  You  : {user_text}")'
)

code = code.replace(
    'logger.debug(f"  Eleven: {response}")',
    'logger.info(f"  Eleven: {response}")'
)

code = code.replace(
    'logger.debug(f"  ⚠️  Suspicious transcript: {reason}")',
    'logger.warning(f"  ⚠️  Suspicious transcript: {reason}")'
)

# For PERF, we want to replace the whole fast_path block and standard block
code = code.replace(
    'logger.debug(f"[PERF] Fast Path:',
    'logger.info(f"\\n[PERF] Total: {t_total:.1f}s\\n")\n                logger.debug(f"[PERF] Fast Path:'
)

# Wait, there are two of those. One for fast path, one for standard.
# Fast path one:
# logger.debug(f"[PERF] Fast Path: {t_fast_path:.0f}ms  ({fp_tag})")
# Standard one:
# logger.debug(f"[PERF] Fast Path: {t_fast_path:.0f}ms  (miss)")

code = code.replace(
    'logger.debug("\\n\\n👋  Goodbye.\\n")',
    'logger.info("\\n\\n👋  Goodbye.\\n")'
)

with open("main.py", "w") as f:
    f.write(code)

