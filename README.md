r"\+?\d[\d\s().-]{6,}\d"
"ADDRESS": re.compile(
        r"\b\d{1,5}\s+(?:[A-Za-z0-9]+\s?){1,5}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Pky|Parkway|Way|Court|Ct|Place|Pl|Circle|Cir)\b.*?(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)?",
        flags=re.IGNORECASE
    )
}
