import re
from transformers import pipeline

# ------------------------------------------------------------------
#  Load a biomedical NER model once (for diagnosis fallback only)
# ------------------------------------------------------------------
ner_pipeline = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",  # You can swap this for a different clinical NER
    grouped_entities=True
)

# ------------------------------------------------------------------
# 1️⃣ Pure regex-based extractor
# ------------------------------------------------------------------
def extract_with_regex(text: str) -> dict:
    """Extract tumor size, stage, and diagnosis using regex only."""
    tumor_size_match = re.search(r'(\d+(\.\d+)?)\s*cm', text)
    stage_match = re.search(r'stage\s([A-Z]+\d*)', text, re.IGNORECASE)
    diagnosis_match = re.search(r'Diagnosis:\s*(.+)', text, re.IGNORECASE)

    return {
        "tumor_size_cm": float(tumor_size_match.group(1)) if tumor_size_match else None,
        "stage": stage_match.group(1).upper() if stage_match else "Unknown",
        "diagnosis": diagnosis_match.group(1).strip() if diagnosis_match else "N/A"
    }

# ------------------------------------------------------------------
# 2️⃣ Hybrid: regex + Hugging Face fallback for diagnosis
# ------------------------------------------------------------------
def extract_with_huggingface(text: str) -> dict:
    tumor_size_match = re.search(r'(\d+(\.\d+)?)\s*cm', text)
    tumor_size = float(tumor_size_match.group(1)) if tumor_size_match else None

    stage_match = re.search(r'stage\s([A-Z]+\d*)', text, re.IGNORECASE)
    stage = stage_match.group(1).upper() if stage_match else "Unknown"

    diagnosis_match = re.search(r'Diagnosis:\s*(.+)', text, re.IGNORECASE)
    diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else "N/A"

    if diagnosis == "N/A":
        ner_results = ner_pipeline(text)
        for ent in ner_results:
            label = ent.get("entity_group", "").lower()
            word = ent.get("word", "").strip()
            if label in {"disease", "disorder"} or any(
                kw in word.lower() for kw in ("cancer", "carcinoma", "tumor", "sarcoma")
            ):
                diagnosis = word
                break

    return {
        "tumor_size_cm": tumor_size,
        "stage": stage,
        "diagnosis": diagnosis
    }

# ------------------------------------------------------------------
# 3️⃣ Public interface: select method
# ------------------------------------------------------------------
def extract_features_from_text(text: str, method: str = "huggingface") -> dict:
    """
    Extract structured clinical fields from unstructured report.

    Parameters
    ----------
    text : str
        Raw clinical report
    method : str
        'regex' or 'huggingface'

    Returns
    -------
    dict
        Contains tumor_size_cm, stage, diagnosis
    """
    if method == "regex":
        return extract_with_regex(text)
    return extract_with_huggingface(text)
