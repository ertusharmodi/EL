from typing import Tuple

def check_sanity(transcript: str, duration_sec: float, confidence: float) -> Tuple[bool, str]:
    """
    Checks if a transcript is likely a Whisper hallucination.
    Returns (is_suspicious, reason).
    """
    clean_text = transcript.strip().lower()
    if not clean_text:
        return False, ""
        
    words = clean_text.split()
    word_count = len(words)
    
    # 1. Unlikely phrases (Whisper specific hallucinations)
    hallucinations = [
        "i'm also a bird",
        "i am an airplane",
        "thank you for watching",
        "thanks for watching"
    ]
    for phrase in hallucinations:
        if phrase in clean_text:
            return True, f"Contains known hallucination phrase: '{phrase}'"
            
    # 2. Many words compared to audio duration
    if duration_sec > 0:
        words_per_sec = word_count / duration_sec
        # Average conversational speech is ~2.5 words/sec. >5 is extremely fast.
        if words_per_sec > 5.0 and word_count > 4:
            return True, f"Unnatural speech rate ({words_per_sec:.1f} words/sec)"
            
        # Very short recording (< 2 seconds) but transcript contains a full sentence.
        if duration_sec < 2.0 and word_count >= 8:
            return True, f"Too many words ({word_count}) for short duration ({duration_sec:.1f}s)"
            
    # 3. Confidence gate: < 0.75 and transcript length > 4 words
    if confidence < 0.75 and word_count > 4:
        return True, f"Low confidence ({confidence:.2f}) for {word_count} words"
        
    return False, ""
