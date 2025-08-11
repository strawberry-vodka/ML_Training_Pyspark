OSError                                   Traceback (most recent call last)
Cell In[17], line 4
      1 import re
      2 import spacy
----> 4 nlp = spacy.load("en_core_web_lg")
      6 # Custom regex patterns for PII not well-covered by default NER
      7 PII_PATTERNS = {
      8     "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
      9     "PHONE": re.compile(r"\b(?:\+?\d{1,3})?[-. (]?\d{3}[-. )]?\d{3}[-. ]?\d{4}\b"),
     10     "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
     11     "ACCOUNT_NUMBER": re.compile(r"\b\d{9,18}\b"),  # Bank account numbers
     12 }

File ~/miniconda3/envs/coa_ai/lib/python3.12/site-packages/spacy/__init__.py:52, in load(name, vocab, disable, enable, exclude, config)
     28 def load(
     29     name: Union[str, Path],
     30     *,
   (...)     35     config: Union[Dict[str, Any], Config] = util.SimpleFrozenDict(),
     36 ) -> Language:
     37     """Load a spaCy model from an installed package or a local path.
     38 
     39     name (str): Package name or model path.
   (...)     50     RETURNS (Language): The loaded nlp object.
     51     """
---> 52     return util.load_model(
     53         name,
     54         vocab=vocab,
     55         disable=disable,
     56         enable=enable,
     57         exclude=exclude,
     58         config=config,
     59     )

File ~/miniconda3/envs/coa_ai/lib/python3.12/site-packages/spacy/util.py:484, in load_model(name, vocab, disable, enable, exclude, config)
    482 if name in OLD_MODEL_SHORTCUTS:
    483     raise IOError(Errors.E941.format(name=name, full=OLD_MODEL_SHORTCUTS[name]))  # type: ignore[index]
--> 484 raise IOError(Errors.E050.format(name=name))

OSError: [E050] Can't find model 'en_core_web_lg'. It doesn't seem to be a Python package or a valid path to a data directory.
