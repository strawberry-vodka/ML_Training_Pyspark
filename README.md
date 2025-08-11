import re
import spacy

# Load spaCy's large English model for better accuracy
nlp = spacy.load("en_core_web_lg")

# Custom regex patterns for PII not well-covered by default NER
PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3})?[-. (]?\d{3}[-. )]?\d{3}[-. ]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ACCOUNT_NUMBER": re.compile(r"\b\d{9,18}\b"),  # Bank account numbers
}

# Entities in spaCy to consider as PII
SPACY_PII_LABELS = {"PERSON", "GPE", "ORG", "LOC", "ADDRESS"}

def mask_pii(text):
    # Step 1: Use spaCy for named entities
    doc = nlp(text)
    masked_text = text

    # Replace detected named entities
    for ent in doc.ents:
        if ent.label_ in SPACY_PII_LABELS:
            masked_text = masked_text.replace(ent.text, f"[{ent.label_}_REDACTED]")

    # Step 2: Apply regex-based PII masking
    for label, pattern in PII_PATTERNS.items():
        masked_text = pattern.sub(f"[{label}_REDACTED]", masked_text)

    return masked_text


# Example transcript
transcript = """
Hi, my name is John Smith. My email is john.smith@example.com and my phone is +1 415-555-2671.
My billing address is 1234 Elm Street, Springfield, IL. 
The credit card number I used was 4111 1111 1111 1111.
"""

masked_transcript = mask_pii(transcript)
print(masked_transcript)
