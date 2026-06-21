import psutil

def execute(user_text: str) -> str:
    """
    Checks if user is asking for system diagnostics.
    Returns None if no match.
    """
    clean = user_text.lower()
    
    triggers = ["memory usage", "ram usage", "cpu usage", "disk usage", "system stats"]
    
    if any(t in clean for t in triggers):
        # We'll just return a combined summary of CPU and RAM for any of these
        # as it's concise and covers the most common metrics.
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        
        # If disk was specifically requested, add it
        if "disk" in clean:
            disk = psutil.disk_usage('/')
            return f"CPU is at {cpu_percent}%, RAM is at {ram_percent}%, and Disk is {disk.percent}% full."
            
        return f"CPU is at {cpu_percent}% and RAM is at {ram_percent}%."
        
    return None
