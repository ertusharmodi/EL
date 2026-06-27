import re

_V2_INTENTS = [
    (r"\b(how are you(?: doing)?|hows it going|how about you|and you)\b", "I'm good.", False),
    (r"\b(what are you doing|what you doing|what r u doing)\b", "Just talking with you.", False),
    (r"\b(miss you|missing you|missed you)\b", "I've missed you too.", False),
    (r"\b(love you|love ya)\b", "Love you too.", False),
    (r"\b(thanks(?: a lot)?|thank you(?: so much| very much)?|appreciate it|many thanks)\b", "You're welcome.", False),
    (r"\b(hi|hello|hey|good morning|good afternoon|good evening|good night)\b", "Hi there!", False),
    (r"\b(bye|goodbye|see you(?: later| soon)?|talk to you later|good bye|take care)\b", "See you later.", True),
    (r"\bi'?m\s+(?:also\s+)?(?:good|fine|great|okay|doing well)\b|\b(?:not bad|pretty good)\b", "Glad to hear that.", False),
    (r"\bi'?m\s+(?:also\s+)?(?:tired|exhausted)\b", "Get some rest.", False),
    (r"\b(okay|ok|alright|got it|sure|sounds good|perfect|nice|cool|great|awesome|makes sense|understood)\b", "Alright.", False),
]

# We need to add "are" to filler, but maybe "how are you" handles it.
_FILLER_WORDS = re.compile(r"\b(and|but|also|just|so|i|im|i'm|am|are|you|baby|eleven|man|bro|dude|well|then|too|a|the|is|my|very|much|so much|a lot|very much|it)\b")

def check(text):
    clean = re.sub(r"[^\w\s']", " ", text.lower())
    matches = []
    for pattern_str, resp, sleep in _V2_INTENTS:
        for match in re.finditer(pattern_str, clean):
            matches.append((match.start(), match.end(), resp, sleep))
            
    if not matches:
        return None, False
        
    matches.sort(key=lambda x: x[0])
    
    filtered_matches = []
    last_end = -1
    for m in matches:
        if m[0] >= last_end:
            filtered_matches.append(m)
            last_end = m[1]
            
    remaining_text = clean
    for start, end, _, _ in reversed(filtered_matches):
        remaining_text = remaining_text[:start] + " " + remaining_text[end:]
        
    remaining_text = _FILLER_WORDS.sub(" ", remaining_text)
    
    if remaining_text.strip():
        return None, False
        
    responses = []
    should_sleep = False
    for _, _, resp, sleep in filtered_matches:
        if resp not in responses:
            responses.append(resp)
        if sleep:
            should_sleep = True
            
    return " ".join(responses), should_sleep

test_cases = [
    "How are you? What are you doing?",
    "How are you and what are you doing?",
    "I'm good but I miss you.",
    "I'm also good.",
    "Thanks baby.",
    "I'm good but turn off the lights."
]

for t in test_cases:
    print(f"'{t}' -> {check(t)}")
