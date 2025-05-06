import os
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY") or "AIzaSyAZ5BSEUTGEOrKeX2AIUdD-CIDuH5lTB1U")

# Load the Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

def generate_treatment_plan(tumor_size_cm, stage, diagnosis):
    if not tumor_size_cm or not stage or diagnosis == "N/A":
        return "Insufficient data to generate a treatment plan. 🛠"

    # Create prompt
    prompt = (
        f"You are a clinical oncologist.\n"
        f"Patient has a {tumor_size_cm} cm tumor, stage {stage}, diagnosed with {diagnosis}.\n"
        "Please recommend a suitable treatment plan in 1–2 sentences based on clinical guidelines."
    )

    try:
        response = model.generate_content(prompt)
        output = response.text.strip()

        # Fallback if output is too short or meaningless
        if len(output) < 20 or diagnosis.lower() in output.lower():
            return rule_based_fallback(tumor_size_cm, stage, diagnosis)

        return output + " 🧠 (Gemini)"

    except Exception as e:
        return f"Gemini API error: {e}"

def rule_based_fallback(tumor_size_cm, stage, diagnosis):
    stage = stage.upper()
    diag = diagnosis.lower()

    if tumor_size_cm >= 5 or stage.startswith("III"):
        return f"Recommend chemotherapy and radiation therapy for {diag}. 🛠"
    elif stage.startswith("I"):
        return f"Surgical resection is preferred for early-stage {diag}. 🛠"
    elif "small cell" in diag:
        return f"Combination chemotherapy is typically used for {diag}. 🛠"
    else:
        return f"Refer to oncology for personalized treatment of {diag}. 🛠"
