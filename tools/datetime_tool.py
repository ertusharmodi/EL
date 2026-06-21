from datetime import datetime

def _ordinal(n):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

def execute(user_text: str) -> str:
    """
    Checks if user is asking for the time or date and returns a conversational string.
    Returns None if no match.
    """
    clean = user_text.lower()
    
    # Check time
    if any(q in clean for q in ("what time", "current time", "the time")):
        now = datetime.now()
        time_str = now.strftime("%-I:%M %p")
        return f"It's {time_str}."
        
    # Check date
    if any(q in clean for q in ("what date", "current date", "today's date", "todays date", "the date", "what is today")):
        now = datetime.now()
        day = _ordinal(now.day)
        date_str = now.strftime(f"%B {day}, %Y")
        return f"Today is {date_str}."
        
    return None
