import streamlit as st
from llm_extraction.extract import extract_features_from_text
from predictive_model.model import predict_survival_risk
from planner.treatment_planner import generate_treatment_plan
from agent.clinical_agent import run_clinical_agent

st.set_page_config(page_title="Clinical AI Assistant", layout="centered")

st.title("🩺 Clinical Report Extractor + AI Treatment Planner")
st.markdown("Paste a clinical report to extract tumor size, stage, and diagnosis, predict survival risk, and generate a treatment plan using Gemini.")

# Input section
report_text = st.text_area("Paste Clinical Report", height=250)

# Extraction method selector
method = st.radio("Extraction Method", ["regex"], index=0)


# On button click
if st.button("Extract & Predict"):
    if not report_text.strip():
        st.warning("Please paste a report.")
    else:
        # --- Step 1: Extraction ---
        result = extract_features_from_text(report_text, method=method)
        st.subheader("🔍 Extracted Fields")
        st.json(result)

        # --- Step 2: Survival risk prediction ---
        st.subheader("📊 Predicted Survival Risk")
        risk = predict_survival_risk(
            tumor_size_cm=result.get("tumor_size_cm"),
            stage=result.get("stage")
        )
        st.success(f"Risk Level: **{risk}**")

        # --- Step 3: Gemini-generated treatment plan ---
        st.subheader("💡 Recommended Treatment Plan (Gemini AI)")
        plan = generate_treatment_plan(
            tumor_size_cm=result.get("tumor_size_cm"),
            stage=result.get("stage"),
            diagnosis=result.get("diagnosis")
        )

        # Display with visual label
        if "🛠" in plan:
            st.info(plan)
        elif "Gemini API error" in plan:
            st.error(plan)
        else:
            st.success(plan)

        # Optional debug: show what method was used
        st.caption(f"Extraction method used: `{method}`")

# Agent mode section
st.header("🧠 Gemini Agent Reasoning")
agent_on = st.checkbox("Use Gemini agent to handle end-to-end report reasoning")

if agent_on and report_text:
    with st.spinner("Thinking with Gemini agent..."):
        result = run_clinical_agent(
            f"Given this report:\n{report_text}\n"
            "1. Extract clinical details.\n"
            "2. Predict survival risk.\n"
            "3. Recommend a treatment plan."
        )
    st.write(result)