#!/usr/bin/env python3
"""Filter ANSI escape codes and unwrap fixed-width text from stdin to stdout."""

import sys
import re

# Remove ANSI escape codes
ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
text = ansi.sub("", sys.stdin.read())

# Replace non-breaking spaces and similar with regular spaces
text = text.replace("\u00A0", " ")  # Non-breaking space
text = text.replace("\u202F", " ")  # Narrow non-breaking space
text = text.replace("\u2007", " ")  # Figure space

# Replace en-dashes and em-dashes with regular hyphens for easier joining
text = text.replace("\u2013", "-")  # En-dash
text = text.replace("\u2014", "-")  # Em-dash

# Handle special hyphen characters that might split words
text = text.replace("\u00AD", "")  # Soft hyphen

# Replace multiple spaces with single space
text = re.sub(r'  +', ' ', text)

lines = text.split("\n")
result = []
i = 0

def should_join(prev, curr):
    """Determine if current line should be joined to previous line."""
    if not prev or not curr:
        return False
    
    prev = prev.strip()
    curr = curr.strip()
    
    if not prev or not curr:
        return False
    
    # Don't join after sentence-ending punctuation
    if prev[-1] in '.!?':
        return False
    
    # Don't join after colon, semicolon - likely new thought
    if prev[-1] in ':;':
        return False
    
    prev_words = prev.split()
    curr_words = curr.split()
    
    if not prev_words or not curr_words:
        return False
    
    last_word = prev_words[-1]
    first_word = curr_words[0]
    
    # If previous line ends with a number and current starts with number or word, join
    if last_word.isdigit():
        return True
    
    # If previous ends with short fragment (1-4 chars), likely cut mid-word
    if len(last_word) <= 4:
        # Join if current starts with lowercase or alphanumeric continuation
        if first_word[0].islower() if first_word else False:
            return True
        if first_word[0].isalnum() if first_word else False:
            return True
    
    # If previous line doesn't end with punctuation and current starts lowercase
    if prev[-1].isalpha():
        if first_word[0].islower():
            return True
    
    # If current starts with lowercase and previous doesn't end sentence
    if first_word[0].islower():
        return True
    
    # If previous line is long (>60 chars) and current is short (<40), join
    if len(prev) > 60 and len(curr) < 40:
        return True
    
    return False

def rejoin_hyphenated(prev, curr):
    """Rejoin words that were split by hyphen at end of previous line."""
    if not prev or not curr:
        return prev + " " + curr if curr else prev
    if prev.endswith('-'):
        return prev[:-1] + curr
    return prev + " " + curr

while i < len(lines):
    line = lines[i].rstrip()

    # Blank line = paragraph break
    if not line:
        if result and result[-1]:
            result.append("")
        i += 1
        continue

    if not result:
        result.append(line)
        i += 1
        continue

    prev = result[-1]
    if not prev:
        result.append(line)
        i += 1
        continue

    # Check for hyphenated word rejoin first
    if prev.endswith('-') and len(line) > 0 and (line[0].isalnum() or line[0].islower()):
        result[-1] = rejoin_hyphenated(prev, line)
        i += 1
        continue

    # Use improved join logic
    if should_join(prev, line):
        result[-1] = prev + " " + line
    else:
        result.append(line)

    i += 1

# Post-process to clean up common artifacts
full_text = "\n".join(result)

# Fix "X X word" pattern (e.g., "36 36 percent" -> "36 percent")
full_text = re.sub(r'(\b\w\b)\s+\1\s+(\w+)', r'\1 \2', full_text)

# Fix "X Y word" where X is short and Y is start of X (e.g., "an and" -> "and")
full_text = re.sub(r'\b(\w{1,3})\s+\1\s+and\b', r'and', full_text)
full_text = re.sub(r'\b(\w{1,3})\s+\1\s+(\w+)', r'\1 \2', full_text)

# Fix "de deploy" -> "deploy" etc.
full_text = re.sub(r'\bde\s+(\w+)\1\b', r'\1', full_text)

# Clean up double spaces
full_text = re.sub(r'  +', ' ', full_text)

print(full_text, end="")