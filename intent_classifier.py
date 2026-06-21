import re

def classify_intent(text: str) -> str:
    """
    Classifies a user message into a predefined intent using lightweight regex and keyword matching.
    Returns strings like 'GREETING', 'THANKS', 'UNKNOWN', etc.
    """
    # Normalize input
    clean_text = text.lower().strip().strip(".!?")
    
    # 1. GOODBYE
    if clean_text in ("bye", "goodbye", "see you", "good night"):
        return "GOODBYE"
        
    # 2. TIME_QUERY
    if clean_text in ("what's time", "what time is it", "tell me time", "current time", "what is the time"):
        return "TIME_QUERY"
        
    # 3. DATE_QUERY
    if clean_text in ("today's date", "current date", "what is today's date", "what is the date"):
        return "DATE_QUERY"
        
    # 4. MEMORY_QUERY
    memory_query_prefixes = (
        "what's my", "what is my", "who is my", "where is my", 
        "when is my", "do you remember"
    )
    if any(clean_text.startswith(prefix) for prefix in memory_query_prefixes):
        return "MEMORY_QUERY"
        
    # 5. MEMORY_UPDATE
    update_prefixes = (
        "remember this", "my favorite", "i like", "i love", 
        "i prefer", "my goal is", "i was born", "my birthplace is"
    )
    if any(clean_text.startswith(prefix) for prefix in update_prefixes):
        return "MEMORY_UPDATE"
        
    # 6. IDENTITY_QUERY
    if clean_text in ("who are you", "what's your name", "what is your name", "what are you doing"):
        return "IDENTITY_QUERY"
        
    # 7. SMALL_TALK (and GREETING/THANKS/ACKNOWLEDGEMENT grouped)
    if clean_text in ("hi", "hello", "hey", "good morning", "good evening"):
        return "GREETING"
    if clean_text in ("thanks", "thank you", "appreciate it"):
        return "THANKS"
    if clean_text in ("okay", "ok", "alright", "got it", "i'll do it", "ill do it", "sounds good"):
        return "ACKNOWLEDGEMENT"
    if clean_text in ("how are you", "how are you doing", "what's up", "how's it going", "i'm good", "im good", "nice", "cool"):
        return "SMALL_TALK"
        
    # 8. LLM_FALLBACK
    return "UNKNOWN"
