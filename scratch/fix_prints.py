import re
import os

files_to_fix = [
    "speech_corrector.py",
    "llm_extractor.py",
    "reminders/manager.py",
    "reminders/storage.py",
    "memory.py",
    "context_resolver.py",
    "llm.py",
    "memory_retriever.py",
    "context_manager.py"
]

for filepath in files_to_fix:
    if not os.path.exists(filepath): continue
    with open(filepath, "r") as f:
        content = f.read()
    
    if "import logger" not in content:
        # Add import logger after other imports
        if "import config" in content:
            content = content.replace("import config\n", "import config\nimport logger\n")
        elif "import json" in content:
            content = content.replace("import json\n", "import json\nimport logger\n")
        else:
            content = "import logger\n" + content
            
    # Warnings
    content = content.replace("print(f\"  ⚠️", "logger.warning(f\"  ⚠️")
    
    # Other prints
    content = content.replace("print(", "logger.debug(")
    
    with open(filepath, "w") as f:
        f.write(content)

