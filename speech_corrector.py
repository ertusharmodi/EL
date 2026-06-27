import logger
import re

CORRECTIONS = {
    "bird": "good",
    "by": "bye",
    "buy": "bye",
    "tusar": "tushar",
    "larvel": "laravel",
    "larval": "laravel",
    "next yes": "nextjs",
    "next jazz": "nextjs",
    "node jay s": "nodejs",
    "react nativee": "react native"
}

def correct_transcript(text: str) -> str:
    """
    Applies common STT corrections using word boundaries and case-insensitive matching.
    """
    original_text = text
    corrected_text = text
    
    for mistake, fix in CORRECTIONS.items():
        # Use regex to match whole words/phrases case-insensitively
        # e.g., \bby\b will match "by" but not "abyssal"
        pattern = r'\b' + re.escape(mistake) + r'\b'
        
        # Count how many times the mistake occurs to log it
        matches = re.findall(pattern, corrected_text, flags=re.IGNORECASE)
        if matches:
            corrected_text = re.sub(pattern, fix, corrected_text, flags=re.IGNORECASE)
            
    if original_text.lower() != corrected_text.lower() and original_text != corrected_text:
        logger.debug(f"  🛠  STT Correction:")
        logger.debug(f"      Before: {original_text}")
        logger.debug(f"      After : {corrected_text}")
        
    return corrected_text
