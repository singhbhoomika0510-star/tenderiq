"""Simple extraction module for Tender‑IQ.

The function `extract_information` receives the raw text of a tender document and returns a dictionary
containing the information requested in the project specification.  The implementation is deliberately
light‑weight and rule‑based – it relies on regular expressions, keyword lists and a tiny TF‑IDF keyword
ranking using scikit‑learn (which is already a transitive dependency of NLTK in this environment).

If a particular piece of information cannot be confidently detected, the value is set to
`"Not found in document"` or `"Needs verification"` as required.
"""

import re
from collections import defaultdict
from typing import Dict, List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure NLTK resources are available – in a real run the user should have downloaded them.
# (The setup instructions include `nltk.download('punkt')` and `nltk.download('stopwords')`.)

STOPWORDS = set(stopwords.words('english'))

# ----- Helper regex patterns -----
DATE_REGEX = re.compile(r"\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b")
MONEY_REGEX = re.compile(r"\b(?:Rs\.?|INR|USD|\$)\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", re.IGNORECASE)
TITLE_REGEX = re.compile(r"(?i)\b(?:Tender\s+Title|Title)\s*[:\-]\s*(.+)")
REF_REGEX = re.compile(r"(?i)\b(?:Reference|Tender\s+No\.?|Tender\s+ID)\s*[:\-]\s*(\S+)")
ORG_REGEX = re.compile(r"(?i)\bIssued\s+by\s*[:\-]\s*(.+)")

# Keyword sets for simple classification
ELIGIBILITY_KEYWORDS = {"experience", "registration", "certification", "qualification", "eligible"}
FINANCIAL_KEYWORDS = {"estimated value", "earnest money", "emd", "tender fee", "performance security", "bank guarantee", "turnover"}
MANDATORY_DOCS_KEYWORDS = {"pan", "gst", "registration certificate", "financial statement", "experience certificate", "technical document"}
DISQUALIFY_KEYWORDS = {"disqualify", "rejection", "not eligible", "late submission", "missing"}

def _extract_first(pattern: re.Pattern, text: str) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else "Not found in document"

def _find_all(pattern: re.Pattern, text: str) -> List[str]:
    return [m.group(0) for m in pattern.finditer(text)]

def _sentence_contains_keywords(sentence: str, keyword_set: set) -> bool:
    tokens = {w.lower() for w in word_tokenize(sentence) if w.isalpha()}
    return bool(tokens & keyword_set)

def _collect_sentences(text: str, keyword_set: set) -> List[str]:
    sentences = sent_tokenize(text)
    return [s for s in sentences if _sentence_contains_keywords(s, keyword_set)]

def _rank_keywords(text: str, top_n: int = 15) -> List[str]:
    # Very lightweight TF‑IDF ranking – split into words, remove stopwords, count frequencies.
    words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in STOPWORDS]
    freq = defaultdict(int)
    for w in words:
        freq[w] += 1
    # Sort by frequency descending
    sorted_words = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]

def extract_information(raw_text: str) -> Dict:
    """Run the rule‑based extraction pipeline.

    Returns a dictionary with the following top‑level keys (matching the specification):
    - overview
    - eligibility
    - financial
    - dates
    - mandatory_docs
    - disqualification
    - alerts
    - keywords
    - summary
    """
    # Basic cleaning – collapse multiple spaces, strip.
    cleaned = re.sub(r"\s+", " ", raw_text).strip()

    # --- Overview ---
    overview = {
        "title": _extract_first(TITLE_REGEX, cleaned),
        "reference": _extract_first(REF_REGEX, cleaned),
        "issuing_organization": _extract_first(ORG_REGEX, cleaned),
        "summary": " ".join(sent_tokenize(cleaned)[:2])  # first two sentences as a brief summary
    }

    # --- Eligibility ---
    eligibility_sentences = _collect_sentences(cleaned, ELIGIBILITY_KEYWORDS)
    eligibility = eligibility_sentences if eligibility_sentences else ["Not found in document"]

    # --- Financial ---
    financial_sentences = _collect_sentences(cleaned, FINANCIAL_KEYWORDS)
    financial = financial_sentences if financial_sentences else ["Not found in document"]

    # --- Dates ---
    dates_found = _find_all(DATE_REGEX, cleaned)
    dates = dates_found if dates_found else ["Not found in document"]

    # --- Mandatory Documents ---
    mandatory_sentences = _collect_sentences(cleaned, MANDATORY_DOCS_KEYWORDS)
    mandatory_docs = mandatory_sentences if mandatory_sentences else ["Not found in document"]

    # --- Disqualification Conditions ---
    disqual_sentences = _collect_sentences(cleaned, DISQUALIFY_KEYWORDS)
    disqualification = disqual_sentences if disqual_sentences else ["Not found in document"]

    # --- Alerts ---
    alerts = []
    # Example alert: multiple different dates that may conflict (simple heuristic)
    if len(dates) > 1:
        alerts.append({
            "type": "Potential date conflict",
            "message": f"Found {len(dates)} date strings – verify they correspond to publication, deadline, etc.",
            "examples": dates[:3]
        })

    # --- Keywords ---
    keywords = _rank_keywords(cleaned)

    # --- Summary ---
    summary = f"Tender '{overview['title']}' (Ref: {overview['reference']}) – extracted {len(eligibility)} eligibility points, {len(financial)} financial points, and {len(dates)} dates."

    return {
        "overview": overview,
        "eligibility": eligibility,
        "financial": financial,
        "dates": dates,
        "mandatory_docs": mandatory_docs,
        "disqualification": disqualification,
        "alerts": alerts,
        "keywords": keywords,
        "summary": summary
    }
