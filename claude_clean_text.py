import re
from typing import Set, List, Tuple

def clean_text(text: str) -> str:
    """
    Enhanced transcription cleaner with improved punctuation handling and
    context-aware repetition detection.
    """
    
    # --- Configuration ---
    
    # Words commonly repeated intentionally for emphasis
    EMPHASIS_WORDS = {
        "no", "yes", "very", "so", "really", "please", "stop", "go", 
        "wait", "help", "oh", "wow", "hey", "hi", "bye"
    }
    
    # Natural reduplications and onomatopoeia
    NATURAL_REPETITIONS = {
        "ha", "haha", "heh", "bye-bye", "night-night", "choo-choo",
        "tick-tock", "ding-dong", "knock-knock", "blah"
    }
    
    # Common filler words that might repeat naturally in speech
    FILLER_WORDS = {"um", "uh", "like", "you know", "I mean"}
    
    # --- Helper Functions ---
    
    def normalize_for_comparison(word: str) -> str:
        """Remove punctuation and lowercase for comparison."""
        return re.sub(r'[^\w\s]', '', word).lower().strip()
    
    def tokenize_preserving_punctuation(text: str) -> List[Tuple[str, str, str]]:
        """
        Tokenize text while preserving punctuation information.
        Returns tuples of (word_with_punct, word_only, punct_only)
        """
        tokens = []
        # Match word with optional trailing punctuation
        pattern = r'(\b\w+\b)([.,!?;:]*)|\s+|([.,!?;:]+)'
        
        for match in re.finditer(pattern, text):
            if match.group(1):  # Word found
                word = match.group(1)
                punct = match.group(2) or ''
                tokens.append((word + punct, word, punct))
            elif match.group(3):  # Standalone punctuation
                tokens.append((match.group(3), '', match.group(3)))
                
        return tokens
    
    # --- Pre-processing ---
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # --- Main Cleaning Rules ---
    
    # Rule 1: Remove extreme repetitions (walls of text)
    # This catches 5+ repetitions of any phrase up to 5 words
    text = re.sub(
        r'((?:\b\S+\b[\s.,!?]*){1,5})\1{4,}',
        r'\1',
        text,
        flags=re.IGNORECASE
    )
    
    # Rule 2: Clean up hyphenated stutters (e.g., "I-I-I-I")
    # Only clean if 3+ repetitions to avoid changing "bye-bye"
    text = re.sub(
        r'\b(\w+)(-\1){2,}\b',
        r'\1',
        text,
        flags=re.IGNORECASE
    )
    
    # Rule 3: Context-aware word repetition cleaning
    tokens = tokenize_preserving_punctuation(text)
    cleaned_tokens = []
    i = 0
    
    while i < len(tokens):
        current_token = tokens[i]
        current_word = current_token[1].lower()
        
        if i + 1 < len(tokens):
            next_token = tokens[i + 1]
            next_word = next_token[1].lower()
            
            # Check if this is a repeated word
            if current_word and current_word == next_word:
                # Determine if we should keep this repetition
                should_keep = False
                
                # Check for emphasis patterns (allow up to 3 repetitions)
                if current_word in EMPHASIS_WORDS:
                    # Count total repetitions
                    rep_count = 1
                    j = i + 1
                    while j < len(tokens) and tokens[j][1].lower() == current_word:
                        rep_count += 1
                        j += 1
                    
                    # Allow up to 3 repetitions for emphasis words
                    if rep_count <= 3:
                        should_keep = True
                
                # Check for natural reduplications
                if current_word in NATURAL_REPETITIONS:
                    should_keep = True
                
                # Check for filler words (allow some repetition)
                if current_word in FILLER_WORDS:
                    # Allow "um um" or "uh uh" but not more
                    if i + 2 >= len(tokens) or tokens[i + 2][1].lower() != current_word:
                        should_keep = True
                
                if not should_keep:
                    # Skip the repetition
                    i += 1
                    while i < len(tokens) and tokens[i][1].lower() == current_word:
                        i += 1
                    # Add only the first occurrence (with its original punctuation)
                    cleaned_tokens.append(current_token[0])
                    continue
        
        cleaned_tokens.append(current_token[0])
        i += 1
    
    # Reconstruct text from tokens
    text = ' '.join(cleaned_tokens)
    
    # Rule 4: Clean up repeated phrases (2-4 words)
    # More conservative - only remove if repeated 3+ times
    text = re.sub(
        r'\b((?:\w+\W+){1,4}\w+)\W+(?:\1\W+){2,}',
        r'\1 ',
        text,
        flags=re.IGNORECASE
    )
    
    # Rule 5: Special case for "you you you" pattern (common Whisper error)
    text = re.sub(
        r'\b(you\s+){3,}you\b',
        'you',
        text,
        flags=re.IGNORECASE
    )
    
    # --- Post-processing ---
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s*([.,!?;:])+', r'\1', text)  # Remove duplicate punctuation
    
    # Normalize whitespace again
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Fix common transcription artifacts
    text = re.sub(r'\.\s*\.\s*\.', '...', text)  # Fix ellipsis
    text = re.sub(r'([!?])\1+', r'\1', text)  # Remove duplicate ! or ?
    
    return text


def analyze_repetitions(text: str) -> dict:
    """
    Analyze text for repetition patterns - useful for debugging.
    """
    analysis = {
        'total_words': len(text.split()),
        'repeated_words': [],
        'repeated_phrases': [],
        'potential_walls': []
    }
    
    # Find repeated words
    words = text.split()
    for i in range(len(words) - 1):
        if i > 0 and words[i].lower() == words[i-1].lower():
            continue  # Skip if already part of a sequence
        
        j = i + 1
        while j < len(words) and words[j].lower() == words[i].lower():
            j += 1
        
        if j - i > 1:
            analysis['repeated_words'].append({
                'word': words[i],
                'count': j - i,
                'position': i
            })
    
    # Find walls of text (5+ repetitions)
    wall_pattern = r'((?:\b\S+\b[\s.,!?]*){1,5})\1{4,}'
    for match in re.finditer(wall_pattern, text, re.IGNORECASE):
        analysis['potential_walls'].append({
            'text': match.group(1)[:50] + '...' if len(match.group(1)) > 50 else match.group(1),
            'position': match.start()
        })
    
    return analysis


# Example usage for testing
if __name__ == "__main__":
    test_cases = [
        # Simple repetitions
        ("I I think we should go", "I think we should go"),
        ("The the the volcano erupted", "The volcano erupted"),
        
        # Emphasis (should be preserved)
        ("No no no! Don't do that!", "No no no! Don't do that!"),
        ("It was very very important", "It was very very important"),
        
        # Walls of text
        ("Thank you. Thank you. Thank you. Thank you. Thank you. Thank you.", "Thank you."),
        ("you you you you you you you should come", "you should come"),
        
        # Hyphenated stutters
        ("I-I-I-I don't know", "I don't know"),
        ("It's a bye-bye situation", "It's a bye-bye situation"),  # Should preserve
        
        # Mixed punctuation
        ("Yeah. Yeah. Yeah. Let's go", "Yeah. Let's go"),
        ("Well, well, well, what do we have here", "Well, what do we have here"),
        
        # Natural speech patterns
        ("Um um I think that uh uh we should", "Um I think that uh we should"),
        ("Ha ha ha that's funny", "Ha ha ha that's funny"),  # Should preserve
        
        # Complex phrases
        ("let's go let's go let's go let's go", "let's go"),
        ("I mean I mean I think", "I mean I think"),
    ]
    
    print("Testing enhanced clean_text function:\n")
    for original, expected in test_cases:
        result = clean_text(original)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{original}'")
        print(f"  → '{result}'")
        if result != expected:
            print(f"  Expected: '{expected}'")
        print()
    
    # Test analysis function
    sample = "The the the cat sat sat on the the mat mat mat mat"
    print("\nAnalyzing sample text:")
    print(f"Original: '{sample}'")
    print(f"Cleaned: '{clean_text(sample)}'")
    print(f"Analysis: {analyze_repetitions(sample)}")