import re

def classify_intent(text: str) -> str:
    """
    Classifies a user message into a predefined intent using lightweight regex and keyword matching.
    Returns strings like 'GREETING', 'THANKS', 'UNKNOWN', etc.
    """
    # Normalize input
    clean_text = text.lower().strip().strip(".!?")
    
    # GREETING
    if clean_text in ("hi", "hello", "hey", "good morning", "good evening"):
        return "GREETING"
        
    # THANKS
    if clean_text in ("thanks", "thank you", "appreciate it"):
        return "THANKS"
        
    # GOODBYE
    if clean_text in ("bye", "goodbye", "see you", "good night"):
        return "GOODBYE"
        
    # IDENTITY
    if clean_text in ("who are you", "what's your name", "what is your name"):
        return "IDENTITY"
        
    # MEMORY_QUERY
    if clean_text in ("what's my name", "what is my name", "what's my favorite color", "what is my favorite color", "do you remember"):
        return "MEMORY_QUERY"
        
    # MEMORY_UPDATE
    update_prefixes = (
        "remember this", "my favorite", "i like", "i love", 
        "i prefer", "my goal is", "i was born", "my birthplace is"
    )
    if any(clean_text.startswith(prefix) for prefix in update_prefixes):
        return "MEMORY_UPDATE"
        
    # SMALL_TALK
    if clean_text in ("how are you", "how are you doing", "what are you doing", "what's up", "how's it going"):
        return "SMALL_TALK"
        
    # ACKNOWLEDGEMENT
    if clean_text in ("okay", "ok", "alright", "got it", "i'll do it", "ill do it", "sounds good"):
        return "ACKNOWLEDGEMENT"
        
    # Future expansion can add explicit QUESTION / COMMAND checks here using NLP heuristics.
    return "UNKNOWN"
