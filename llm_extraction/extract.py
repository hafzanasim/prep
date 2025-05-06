import re

def extract_features_from_text(text: str, method: str = "regex") -> dict:
    """
    Extracts tumor size, cancer stage, and diagnosis from unstructured clinical report text using regular expressions.

    Parameters:
        text (str): The full clinical report.

    Returns:
        dict: {
            "tumor_size_cm": float or None,
            "stage": str (e.g., "IIIA" or "Unknown"),
            "diagnosis": str
        }
    """
    # Match tumor size like "4.2 cm"
    tumor_size_match = re.search(r'(\d+(\.\d+)?)\s*cm', text)

    # Match stage like "Stage IIIA"
    stage_match = re.search(r'stage\s([A-Z]+\d*)', text, re.IGNORECASE)

    # Match diagnosis like "Diagnosis: Non-small cell lung cancer."
    diagnosis_match = re.search(r'Diagnosis:\s*(.+)', text, re.IGNORECASE)

    return {
        "tumor_size_cm": float(tumor_size_match.group(1)) if tumor_size_match else None,
        "stage": stage_match.group(1).upper() if stage_match else "Unknown",
        "diagnosis": diagnosis_match.group(1).strip() if diagnosis_match else "N/A"
    }