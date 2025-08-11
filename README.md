import re
import spacy

# Try loading large model; fallback to small if not installed
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    nlp = spacy.load("en_core_web_sm")

# Updated regex patterns for PII
PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    
    # More flexible phone number detection: catches US & international formats
    "PHONE": re.compile(
        r"\+?\d[\d\s().-]{6,}\d"  # min length ~8 digits, allows symbols
    ),
    
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ACCOUNT_NUMBER": re.compile(r"\b\d{9,18}\b"),
    
    # Expanded address regex
    "ADDRESS": re.compile(
        r"\b\d{1,6}\s+(?:[A-Za-z0-9]+\s?){0,6}"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Pky|Parkway|Way|Court|Ct|Place|Pl|Circle|Cir)"
        r"(?:,\s*[A-Za-z ]+)?(?:,\s*[A-Za-z ]+)?",
        flags=re.IGNORECASE
    )
}

# spaCy PII labels to mask
SPACY_PII_LABELS = {"PERSON", "GPE", "ORG", "LOC", "ADDRESS"}

def mask_pii(text):
    doc = nlp(text)
    masked_text = text

    # Step 1: Mask NER entities
    for ent in doc.ents:
        if ent.label_ in SPACY_PII_LABELS:
            masked_text = re.sub(re.escape(ent.text), f"[{ent.label_}_REDACTED]", masked_text)

    # Step 2: Mask regex-based patterns
    for label, pattern in PII_PATTERNS.items():
        masked_text = pattern.sub(f"[{label}_REDACTED]", masked_text)

    return masked_text


# Example test
transcript = """
Hi, my name is John Smith. My email is john.smith@example.com.
Call me at +1 415-555-2671 or +44 20 7946 0958.
My billing address is 1280 S pky, North California or 45 Main Street, Springfield.
My bank account is 987654321012 and my credit card is 4111 1111 1111 1111.
"""

print(mask_pii(transcript))
