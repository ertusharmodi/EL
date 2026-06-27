import subprocess

def execute(command: str) -> str:
    """
    Launches predefined applications on macOS.
    Supported apps: Cursor, VSCode, Chrome, Spotify, WhatsApp, etc.
    If the command starts with 'open ', it will always attempt to launch and return a string, 
    preventing fallback to the LLM.
    """
    clean_cmd = command.lower().strip()
    
    # Map command keywords to actual macOS application names
    app_map = {
        "cursor": "Cursor",
        "vscode": "Visual Studio Code",
        "chrome": "Google Chrome",
        "spotify": "Spotify",
        "whatsapp": "WhatsApp"
    }
    
    if clean_cmd.startswith("open "):
        target = clean_cmd.replace("open ", "").strip()
        
        # Determine actual app name (either from map or capitalize the target)
        app_name = None
        for key, mapped_name in app_map.items():
            if key in target:
                app_name = mapped_name
                break
                
        if not app_name:
            app_name = target.title()
            
        try:
            subprocess.run(["open", "-a", app_name], check=True, capture_output=True)
            return f"Opening {app_name}."
        except subprocess.CalledProcessError:
            return f"I couldn't open {app_name}."
            
    return None
