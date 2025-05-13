import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent, AgentType

from llm_extraction.extract import extract_features_from_text
from predictive_model.model import predict_survival_risk
from planner.treatment_planner import generate_treatment_plan

# ---------------------------------------------
# Gemini LLM configuration via LangChain
# ---------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY") or "AIzaSyAZ5BSEUTGEOrKeX2AIUdD-CIDuH5lTB1U",
    temperature=0.3
)

# ---------------------------------------------
# Helper function: risk prediction from JSON
# ---------------------------------------------
def _safe_predict_risk(text):
    try:
        data = json.loads(text)
        size = float(data.get("tumor_size_cm", 0))
        stage = data.get("stage", "Unknown")
        return f"Risk: {predict_survival_risk(size, stage)}"
    except Exception as e:
        return f"Error in prediction: {e}"

# ---------------------------------------------
# Helper function: treatment planning from JSON
# ---------------------------------------------
def _safe_generate_plan(text):
    try:
        data = json.loads(text)
        size = float(data.get("tumor_size_cm", 0))
        stage = data.get("stage", "Unknown")
        diagnosis = data.get("diagnosis", "N/A")
        return generate_treatment_plan(size, stage, diagnosis)
    except Exception as e:
        return f"Error generating plan: {e}"

# ---------------------------------------------
# Tool definitions for the agent
# ---------------------------------------------
tools = [
    Tool(
        name="ExtractReport",
        func=lambda text: extract_features_from_text(text, method="gemini"),
        description="Extract tumor size, stage, and diagnosis from a clinical report"
    ),
    Tool(
        name="PredictRisk",
        func=_safe_predict_risk,
        description="Predict survival risk. Expects a JSON string with keys: tumor_size_cm (float), stage (str)"
    ),
    Tool(
        name="GenerateTreatmentPlan",
        func=_safe_generate_plan,
        description="Generate a treatment plan. Expects a JSON string with keys: tumor_size_cm, stage, diagnosis"
    )
]

# ---------------------------------------------
# Agent initialization
# ---------------------------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ---------------------------------------------
# Run the agent with a full prompt
# ---------------------------------------------
def run_clinical_agent(input_text: str) -> str:
    return agent.run(input_text)
