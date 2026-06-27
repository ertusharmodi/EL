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

        
    # 4. MEMORY_QUERY
    memory_query_prefixes = (
        "what's my", "what is my", "who is my", "where is my", 
        "when is my", "do you remember"
    )
    if any(clean_text.startswith(prefix) for prefix in memory_query_prefixes):
        return "MEMORY_QUERY"
        
    # 5. MEMORY_SUMMARY
    summary_prefixes = (
        "what do you know about me", "list my preferences", "summarize my memory", "what is my summary",
        "do you know me", "do you know who i am", "who am i"
    )
    if clean_text in summary_prefixes or any(clean_text.startswith(prefix) for prefix in summary_prefixes):
        return "MEMORY_SUMMARY"
        
    # 6. MEMORY_REMEMBER
    remember_prefixes = (
        "remember that", "remember this", "my goal is", "i was born", "my birthplace is"
    )
    if any(clean_text.startswith(prefix) for prefix in remember_prefixes):
        return "MEMORY_REMEMBER"
        
    # 7. MEMORY_UPDATE
    update_prefixes = (
        "update my", "change my", "my favorite", "i like", "i love", "i prefer"
    )
    if any(clean_text.startswith(prefix) for prefix in update_prefixes):
        return "MEMORY_UPDATE"
        
    # 8. MEMORY_FORGET
    forget_prefixes = (
        "forget my", "forget that", "delete my"
    )
    if any(clean_text.startswith(prefix) for prefix in forget_prefixes):
        return "MEMORY_FORGET"
        
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
