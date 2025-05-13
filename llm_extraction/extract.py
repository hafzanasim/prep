import os
import json
import re
import google.generativeai as genai

# Configure Gemini with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY") or "AIzaSyAZ5BSEUTGEOrKeX2AIUdD-CIDuH5lTB1U")

def generate_gemini_prompt(report_text: str) -> str:
    return f'''
You are a clinical language model. Extract the following structured fields from the clinical report below:

- patient_id: name, age, sex, empi_id, cmrn, fin_number, account_number
- vitals: pulse, heart_rate, temp, blood_sugar
- tumor_size_cm
- stage
- diagnosis
- medical_history (as a list)
- symptoms (as a list): e.g., ["nausea", "fatigue"]
- medications
- last_visit_summary
- radiology: modality, organ, tumor_type, summary, past_findings
- medical_findings

Respond ONLY with valid JSON. Do not include extra text or commentary.

Report:
"""
{report_text}
"""
'''

def extract_json_from_response(text: str) -> dict:
    match = re.search(r'{.*}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise json.JSONDecodeError("No JSON object found", text, 0)

def extract_features_from_text(text: str, method: str = "gemini") -> dict:
    """
    Extracts structured clinical features from a medical report using Gemini LLM.

    Parameters:
        text (str): The clinical report.
        method (str): Extraction method, default is "gemini".

    Returns:
        dict: Structured clinical information.
    """
    if method != "gemini":
        raise ValueError("Only 'gemini' method is currently supported in this version.")

    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    prompt = generate_gemini_prompt(text)
    response = model.generate_content(prompt)

    try:
        return extract_json_from_response(response.text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response.", "raw_output": response.text}
