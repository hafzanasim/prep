import streamlit as st
from streamlit_lottie import st_lottie
import requests
import sqlite3
from datetime import datetime
import json
import pytz

from llm_extraction.extract import extract_features_from_text
from predictive_model.model import predict_survival_risk
from planner.treatment_planner import generate_treatment_plan
from agent.clinical_agent import run_clinical_agent

# ------------------- Utility Functions -------------------

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def init_db():
    conn = sqlite3.connect("oncology_reports.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        sex TEXT,
        empi_id TEXT,
        tumor_size REAL,
        stage TEXT,
        diagnosis TEXT,
        survival_risk TEXT,
        is_high_risk INTEGER,
        report TEXT,
        extracted_json TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def save_report(data, report_text, risk):
    is_high_risk = 1 if risk and "High" in risk else 0
    conn = sqlite3.connect("oncology_reports.db")
    cursor = conn.cursor()

    pid = data.get("patient_id", {})
    name = pid.get("name")
    age = pid.get("age")
    sex = pid.get("sex")
    empi_id = pid.get("empi_id")

    cursor.execute("""
        SELECT 1 FROM reports WHERE 
        name = ? AND age = ? AND sex = ? AND report = ?
    """, (name, age, sex, report_text))

    if cursor.fetchone() is None:
        cursor.execute("""INSERT INTO reports
            (name, age, sex, empi_id, tumor_size, stage, diagnosis, survival_risk, is_high_risk, report, extracted_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, age, sex, empi_id,
            data.get("tumor_size_cm"), data.get("stage"), data.get("diagnosis"),
            risk, is_high_risk, report_text, json.dumps(data)
        ))
        conn.commit()
    conn.close()

def display_full_json(data):
    pid = data.get("patient_id", {})
    st.markdown("### 👤 Patient Info")
    for key in ["name", "age", "sex", "empi_id", "cmrn", "fin_number", "account_number"]:
        st.write(f"**{key.replace('_',' ').title()}**: {pid.get(key)}")

    st.markdown("### 🧠 Tumor & Diagnosis")
    st.write(f"**Tumor Size (cm):** {data.get('tumor_size_cm')}")
    st.write(f"**Stage:** {data.get('stage')}")
    st.write(f"**Diagnosis:** {data.get('diagnosis')}")

    st.markdown("### 🩺 Vitals")
    for k, v in data.get("vitals", {}).items():
        st.write(f"**{k.title().replace('_',' ')}:** {v}")

    st.markdown("### 📚 Medical History & Symptoms")
    history = data.get("medical_history")
    if isinstance(history, str):
        history = [h.strip() for h in history.split(",") if h.strip()]
    elif not history:
        history = []

    st.write("**History:**", ", ".join(history) or "None")
    st.write("**Symptoms:**", ", ".join(data.get("symptoms") or []) or "None")

    st.markdown("### 💊 Medications")
    meds = data.get("medications")
    st.write(", ".join(meds) if isinstance(meds, list) else meds or "None")

    st.markdown("### 📝 Last Visit Summary")
    st.write(data.get("last_visit_summary") or "None")

    st.markdown("### 🧪 Radiology")
    radiology = data.get("radiology")

    if isinstance(radiology, dict):
        for k in ["modality", "organ", "tumor_type", "summary", "past_findings"]:
            st.write(f"**{k.replace('_',' ').title()}:** {radiology.get(k)}")

    elif isinstance(radiology, list):
        for idx, entry in enumerate(radiology, 1):
            st.markdown(f"**🧪 Entry {idx}**")
            for k, v in entry.items():
                st.write(f"**{k.replace('_',' ').title()}:** {v}")
    else:
        st.write("No radiology data available.")

    st.markdown("### 🧾 Medical Findings")
    findings = data.get("medical_findings")
    st.write(", ".join(findings) if isinstance(findings, list) else findings or "None")

# ------------------- App Configuration -------------------

init_db()

st.set_page_config(page_title="🧬 Oncology Agent App", layout="wide")

pineapple_animation = load_lottie_url("https://lottie.host/f8598c5c-0553-4d15-b6d1-1c688a31cbd3/sPxmgjFZxG.json")
if pineapple_animation:
    st_lottie(pineapple_animation, height=150, speed=1, loop=True)

st.title("🩺 Oncology Agent App")
st.sidebar.title("🧬 Navigation")

section = st.sidebar.radio("Jump to:", ["📑 Extract & Summarize", "📊 Risk Dashboard", "📘 About"])

# ------------------- Section: Extract & Lookup -------------------

if section == "📑 Extract & Summarize":
    tab1, tab2 = st.tabs(["📝 New Report", "🔎 Patient Lookup"])

    with tab1:
        st.markdown("## Paste Clinical Report")
        report_text = st.text_area("Input your clinical report:", height=200)
        if st.button("🔍 Extract"):
            if not report_text.strip():
                st.warning("Please paste a clinical report.")
            else:
                data = extract_features_from_text(report_text)
                if "error" in data:
                    st.error("Gemini parsing failed. Showing raw output:")
                    st.text(data.get("raw_output"))
                else:
                    st.success("Extraction complete!")
                    pid = data.get("patient_id", {})
                    st.subheader(f"🧍 {pid.get('name', 'Unknown')} | Age {pid.get('age', '?')}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Tumor Size (cm)", data.get("tumor_size_cm"))
                    col2.metric("Stage", data.get("stage"))
                    risk = predict_survival_risk(data.get("tumor_size_cm"), data.get("stage"))
                    col3.metric("Predicted Risk", risk)

                    display_full_json(data)

                    plan = generate_treatment_plan(data.get("tumor_size_cm"), data.get("stage"), data.get("diagnosis"))
                    st.info(plan)
                    save_report(data, report_text, risk)

    with tab2:
        st.markdown("## Enter EMPI ID to Look Up Record")
        empi = st.text_input("EMPI ID")
        if st.button("Search EMPI"):
            conn = sqlite3.connect("oncology_reports.db")
            cursor = conn.cursor()
            cursor.execute("SELECT extracted_json, timestamp FROM reports WHERE empi_id = ?", (empi,))
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                st.warning("No record found.")
            else:
                eastern = pytz.timezone("America/New_York")
                for json_blob, ts in rows:
                    local_time = (
                        datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                        .astimezone(eastern)
                        .strftime("%Y-%m-%d %H:%M:%S")
                    )
                    st.markdown(f"### 📄 Record from {local_time}")
                    data = json.loads(json_blob)
                    display_full_json(data)

# ------------------- Section: Risk Dashboard -------------------

elif section == "📊 Risk Dashboard":
    st.markdown("## 🚨 High-Risk Patients")
    conn = sqlite3.connect("oncology_reports.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, age, stage, survival_risk, extracted_json, timestamp
        FROM reports WHERE is_high_risk = 1 ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        st.info("No high-risk records.")
    else:
        eastern = pytz.timezone("America/New_York")
        for name, age, stage, risk, json_blob, ts in rows:
            local_time = (
                datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                .astimezone(eastern)
                .strftime("%Y-%m-%d %H:%M:%S")
            )
            with st.expander(f"🧬 {name}, Age {age} — Stage {stage} — {risk}"):
                st.caption(f"📅 Recorded: {local_time}")
                data = json.loads(json_blob)
                display_full_json(data)

# ------------------- Section: About -------------------

elif section == "📘 About":
    st.markdown("## 📘 About This App")
    st.markdown("""
    This app uses Google's Gemini to extract structured information from oncology reports, estimate survival risk,
    and generate treatment suggestions. Features:

    - Report extraction & summarization  
    - High-risk patient dashboard  
    - EMPI-based patient record lookup  
    """)
