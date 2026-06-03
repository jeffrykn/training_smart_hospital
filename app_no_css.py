import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Smart Hospital Patient Navigator",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("hospital_model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_model()

model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
cols_to_scale = bundle["cols_to_scale"]
dept_map_inv = bundle["dept_map_inv"]

gender_map = bundle["gender_map"]
temp_map = bundle["temp_map"]
hr_map = bundle["hr_map"]
dur_map = bundle["dur_map"]
cc_map = bundle["cc_map"]

# -----------------------------
# Department Information
# -----------------------------
DEPT_INFO = {
    "Respiratory Medicine": {
        "icon": "🫁",
        "desc": "Specialises in conditions affecting the lungs and airways.",
        "next": [
            "Visit Level 2, Wing B",
            "Estimated wait: 15–25 min",
            "Please wear a mask"
        ]
    },
    "Cardiology": {
        "icon": "❤️",
        "desc": "Specialises in heart and cardiovascular conditions.",
        "next": [
            "Visit Level 3, Wing A",
            "Estimated wait: 20–30 min",
            "Bring any previous ECG reports"
        ]
    },
    "Gastroenterology": {
        "icon": "🫃",
        "desc": "Specialises in digestive system and abdominal conditions.",
        "next": [
            "Visit Level 1, Wing C",
            "Estimated wait: 10–20 min",
            "Avoid eating before consultation"
        ]
    },
    "Neurology": {
        "icon": "🧠",
        "desc": "Specialises in brain, spine, and nervous system conditions.",
        "next": [
            "Visit Level 4, Wing A",
            "Estimated wait: 25–35 min",
            "Bring list of current medications"
        ]
    },
    "General Medicine": {
        "icon": "🩺",
        "desc": "Handles general health concerns and non-specialist conditions.",
        "next": [
            "Visit Level 1, Wing A",
            "Estimated wait: 10–15 min",
            "Registration desk is open 24/7"
        ]
    },
    "Dermatology": {
        "icon": "🔬",
        "desc": "Specialises in skin, hair, and nail conditions.",
        "next": [
            "Visit Level 2, Wing D",
            "Estimated wait: 15–20 min",
            "Bring photos of affected area if possible"
        ]
    },
}

# -----------------------------
# Header
# -----------------------------
st.title("🏥 Smart Hospital Patient Navigator")
st.subheader("Find the Right Department for Your Symptoms")
st.caption("Future Classroom · Machine Learning")

st.divider()

# -----------------------------
# Form
# -----------------------------
with st.form("triage_form"):

    st.header("1. What are your main symptoms?")
    st.caption("Select all that apply")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        fever = st.checkbox("🌡️ Fever")
        cough = st.checkbox("🤧 Cough")

    with c2:
        headache = st.checkbox("🤕 Headache")
        chest_pain = st.checkbox("💔 Chest Pain")

    with c3:
        stomach_pain = st.checkbox("🤢 Stomach Pain")
        shortness_breath = st.checkbox("😮‍💨 Shortness of Breath")

    with c4:
        nausea_vomiting = st.checkbox("🤮 Nausea / Vomiting")
        dizziness = st.checkbox("😵 Dizziness")

    skin_rash = st.checkbox("🔴 Skin Rash")

    st.divider()

    st.header("2. How long have you had these symptoms?")

    col1, col2 = st.columns(2)

    with col1:
        chief_complaint = st.selectbox(
            "Chief Complaint",
            options=list(cc_map.keys())
        )

    with col2:
        duration = st.selectbox(
            "Duration",
            options=list(dur_map.keys()),
            index=1
        )

    st.divider()

    st.header("3. Severity")

    col1, col2 = st.columns(2)

    with col1:
        temperature_level = st.selectbox(
            "Temperature",
            options=list(temp_map.keys()),
            index=1
        )

    with col2:
        heart_rate_level = st.selectbox(
            "Heart Rate",
            options=list(hr_map.keys()),
            index=1
        )

    st.divider()

    st.header("4. Medical History")

    col1, col2, col3 = st.columns(3)

    with col1:
        hypertension = st.checkbox("🩺 High Blood Pressure")

    with col2:
        heart_disease = st.checkbox("❤️ Heart Disease")

    with col3:
        asthma = st.checkbox("💨 Asthma")

    st.divider()

    st.header("5. Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=35
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    submitted = st.form_submit_button("Get AI Recommendation")

# -----------------------------
# Prediction
# -----------------------------
if submitted:

    patient = pd.DataFrame([{
        "age": age,
        "gender": gender_map.get(gender, 0),

        "fever": int(fever),
        "cough": int(cough),
        "headache": int(headache),
        "chest_pain": int(chest_pain),
        "stomach_pain": int(stomach_pain),
        "shortness_breath": int(shortness_breath),
        "nausea_vomiting": int(nausea_vomiting),
        "dizziness": int(dizziness),
        "skin_rash": int(skin_rash),

        "temperature_level": temp_map.get(temperature_level, 1),
        "heart_rate_level": hr_map.get(heart_rate_level, 1),
        "duration": dur_map.get(duration, 1),

        "asthma": int(asthma),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),

        "chief_complaint": cc_map.get(chief_complaint, 9)
    }])

    patient_scaled = patient.copy()

    patient_scaled[cols_to_scale] = scaler.transform(
        patient[cols_to_scale]
    )

    pred = model.predict(patient_scaled[features])[0]
    proba = model.predict_proba(patient_scaled[features])[0]

    dept_name = dept_map_inv[pred]
    confidence = proba[pred] * 100

    info = DEPT_INFO[dept_name]

    st.divider()

    st.header("AI Recommendation")

    st.success(
        f"Recommended Department: {info['icon']} {dept_name}"
    )

    st.write(info["desc"])

    st.write(f"**Confidence:** {confidence:.1f}%")

    st.subheader("Next Steps")

    for step in info["next"]:
        st.write(f"• {step}")

    st.warning(
        "This is an AI suggestion, not a medical diagnosis. "
        "Please consult a doctor for further evaluation."
    )

    st.divider()

    st.subheader("Confidence by Department")

    sorted_depts = sorted(
        dept_map_inv.items(),
        key=lambda x: proba[x[0]],
        reverse=True
    )

    for idx, dname in sorted_depts:
        pct = float(proba[idx])

        st.write(
            f"{DEPT_INFO[dname]['icon']} "
            f"{dname} — {pct * 100:.1f}%"
        )

        st.progress(pct)

    st.info(
        "Model: KNN (k=7)\n\n"
        "Dataset: 102,000 patients\n\n"
        "Accuracy: 99.5%\n\n"
        "Powered by Future Classroom ML"
    )
