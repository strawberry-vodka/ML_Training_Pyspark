import re
import spacy
from pathlib import Path

# Load spaCy small model with only NER enabled
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
nlp.max_length = 2_000_000  # Safety for long transcripts

# Compile regex patterns once
PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\+?\d[\d\s().-]{6,}\d"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ACCOUNT_NUMBER": re.compile(r"\b\d{9,18}\b"),
    "ADDRESS": re.compile(
        r"\b\d{1,6}\s+(?:[A-Za-z0-9]+\s?){0,6}"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Pky|Parkway|Way|Court|Ct|Place|Pl|Circle|Cir)"
        r"(?:,\s*[A-Za-z ]+)?(?:,\s*[A-Za-z ]+)?",
        flags=re.IGNORECASE
    )
}

SPACY_PII_LABELS = {"PERSON", "GPE", "ORG", "LOC", "ADDRESS"}


def mask_text(text, ents):
    """Mask both spaCy entities and regex matches."""
    masked_text = text
    # Mask spaCy entities
    for ent_text, label in ents:
        masked_text = re.sub(re.escape(ent_text), f"[{label}_REDACTED]", masked_text)
    # Mask regex matches
    for label, pattern in PII_PATTERNS.items():
        masked_text = pattern.sub(f"[{label}_REDACTED]", masked_text)
    return masked_text


def mask_pii_batch(texts):
    """Process transcripts in batches for speed."""
    masked_texts = []
    for doc in nlp.pipe(texts, batch_size=50):  # Larger batch size = faster
        ents = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in SPACY_PII_LABELS]
        masked_texts.append(mask_text(doc.text, ents))
    return masked_texts


# Example usage on folder of transcripts
input_folder = Path("transcripts_raw")
output_folder = Path("transcripts_masked")
output_folder.mkdir(exist_ok=True)

# Read all transcripts
texts = [p.read_text(encoding="utf-8") for p in input_folder.glob("*.txt")]

# Mask in batch
masked_results = mask_pii_batch(texts)

# Save results
for i, p in enumerate(input_folder.glob("*.txt")):
    (output_folder / p.name).write_text(masked_results[i], encoding="utf-8")

print(f"Masked {len(texts)} transcripts successfully!")
