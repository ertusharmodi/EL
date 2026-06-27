import retriever
from typing import Tuple, Optional

def _sanitise(text: str) -> str:
    """Helper to cleanly evaluate user text for small differences."""
    return text.lower().strip().strip(".!?")

def apply_policy(intent: str, user_text: str) -> Tuple[Optional[str], bool]:
    """
    Applies the response policy for a given intent.
    Returns:
        response (str | None): The hardcoded response, or None if it should fall through to the LLM.
        should_sleep (bool): True if the assistant should enter sleep mode.
    """
    clean_msg = _sanitise(user_text)
    should_sleep = False
    response = None

    if intent == "MEMORY_REMEMBER":
        response = "Got it."
        
    elif intent == "MEMORY_UPDATE":
        response = "Updated."
        
    elif intent == "MEMORY_FORGET":
        response = "Forgotten."
        
    elif intent == "MEMORY_SUMMARY":
        import memory_manager
        response = memory_manager.get_summary()
        
    elif intent == "MEMORY_QUERY":
        ans = retriever.retrieve_direct_answer(user_text)
        if ans:
            response = ans
            
    elif intent == "GOODBYE":
        response = "See you later."
        should_sleep = True
        
    elif intent == "THANKS":
        response = "You're welcome."

        
    elif intent == "IDENTITY_QUERY":
        if clean_msg == "who are you":
            response = "I'm Eleven."
        elif clean_msg in ("what's your name", "what is your name"):
            response = "Eleven."
        elif clean_msg == "what are you doing":
            response = "Just talking with you."
            
    elif intent == "GREETING":
        if "good morning" in clean_msg:
            response = "Good morning."
        else:
            response = "Hi."
            
    elif intent == "SMALL_TALK":
        if clean_msg in ("i'm good", "im good"):
            response = "Good to hear."
        elif clean_msg in ("nice", "cool"):
            response = "Nice."
        elif "how are you" in clean_msg or "how's it going" in clean_msg:
            response = "I'm good. How about you?"
        else:
            response = "Alright."
            
    elif intent == "ACKNOWLEDGEMENT":
        if "i'll do it" in clean_msg or "ill do it" in clean_msg or clean_msg == "sounds good":
            response = "Sounds good."
        else:
            response = "Alright."
            
    return response, should_sleep
