import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Smart Hospital Patient Navigator", page_icon="🏥", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none; }
.stDeployButton { display: none; }
footer { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 1100px !important; }
div[data-testid="stForm"] { border: none; padding: 0; }

div.stButton > button {
    background: linear-gradient(135deg, #1a56db, #1e429f) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.75rem 2rem !important;
    font-size: 16px !important; font-weight: 600 !important;
    width: 100% !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(26,86,219,0.35) !important;
}
div.stButton > button:hover { background: linear-gradient(135deg, #1e429f, #1a56db) !important; }

div[data-testid="stCheckbox"] label {
    font-size: 14px !important; font-weight: 500 !important; color: #374151 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open('hospital_model.pkl', 'rb') as f:
        return pickle.load(f)

bundle        = load_model()
model         = bundle['model']
scaler        = bundle['scaler']
features      = bundle['features']
cols_to_scale = bundle['cols_to_scale']
dept_map_inv  = bundle['dept_map_inv']
gender_map    = bundle['gender_map']
temp_map      = bundle['temp_map']
hr_map        = bundle['hr_map']
dur_map       = bundle['dur_map']
cc_map        = bundle['cc_map']

DEPT_INFO = {
    'Respiratory Medicine': {
        'icon':'🫁','color':'#0284c7','bg':'#e0f2fe','border':'#7dd3fc',
        'desc':'Specialises in conditions affecting the lungs and airways.',
        'next':['Visit Level 2, Wing B','Estimated wait: 15–25 min','Please wear a mask']
    },
    'Cardiology': {
        'icon':'❤️','color':'#dc2626','bg':'#fee2e2','border':'#fca5a5',
        'desc':'Specialises in heart and cardiovascular conditions.',
        'next':['Visit Level 3, Wing A','Estimated wait: 20–30 min','Bring any previous ECG reports']
    },
    'Gastroenterology': {
        'icon':'🫃','color':'#d97706','bg':'#fef3c7','border':'#fcd34d',
        'desc':'Specialises in digestive system and abdominal conditions.',
        'next':['Visit Level 1, Wing C','Estimated wait: 10–20 min','Avoid eating before consultation']
    },
    'Neurology': {
        'icon':'🧠','color':'#7c3aed','bg':'#ede9fe','border':'#c4b5fd',
        'desc':'Specialises in brain, spine, and nervous system conditions.',
        'next':['Visit Level 4, Wing A','Estimated wait: 25–35 min','Bring list of current medications']
    },
    'General Medicine': {
        'icon':'🩺','color':'#059669','bg':'#d1fae5','border':'#6ee7b7',
        'desc':'Handles general health concerns and non-specialist conditions.',
        'next':['Visit Level 1, Wing A','Estimated wait: 10–15 min','Registration desk is open 24/7']
    },
    'Dermatology': {
        'icon':'🔬','color':'#b45309','bg':'#fef9c3','border':'#fde68a',
        'desc':'Specialises in skin, hair, and nail conditions.',
        'next':['Visit Level 2, Wing D','Estimated wait: 15–20 min','Bring photos of affected area if possible']
    },
}

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1e3a8a 0%,#1a56db 60%,#0ea5e9 100%);
            padding:3rem 2rem 2.5rem;margin:-1rem -1rem 2rem;text-align:center;">
    <div style="font-size:14px;font-weight:500;color:rgba(255,255,255,0.7);
                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">
        🏥 Future Classroom · Machine Learning
    </div>
    <div style="font-size:36px;font-weight:700;color:#ffffff;margin-bottom:12px;
                letter-spacing:-0.02em;">
        Smart Hospital Patient Navigator
    </div>
    <div style="font-size:18px;color:rgba(255,255,255,0.85);font-weight:400;">
        Find the Right Department for Your Symptoms
    </div>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("triage_form"):

    # Section 1 — Symptoms
    st.markdown("""
    <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="background:#0284c7;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">1</span>
            <span style="font-size:16px;font-weight:600;color:#0c4a6e;">What are your main symptoms?</span>
            <span style="font-size:13px;color:#6b7280;font-style:italic;">select all that apply</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fever            = st.checkbox("🌡️  Fever")
        cough            = st.checkbox("🤧  Cough")
    with c2:
        headache         = st.checkbox("🤕  Headache")
        chest_pain       = st.checkbox("💔  Chest Pain")
    with c3:
        stomach_pain     = st.checkbox("🤢  Stomach Pain")
        shortness_breath = st.checkbox("😮‍💨  Shortness of Breath")
    with c4:
        nausea_vomiting  = st.checkbox("🤮  Nausea / Vomiting")
        dizziness        = st.checkbox("😵  Dizziness")

    c5, _, _, _ = st.columns(4)
    with c5:
        skin_rash = st.checkbox("🔴  Skin Rash")

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2 — Duration & Complaint
    st.markdown("""
    <div style="background:#fdf4ff;border:1px solid #e9d5ff;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#7c3aed;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">2</span>
            <span style="font-size:16px;font-weight:600;color:#3b0764;">How long have you had these symptoms?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cc, col_dur = st.columns(2)
    with col_cc:
        chief_complaint = st.selectbox("Chief complaint", options=list(cc_map.keys()))
    with col_dur:
        duration = st.selectbox("Duration", options=list(dur_map.keys()), index=1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 3 — Severity
    st.markdown("""
    <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#ea580c;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">3</span>
            <span style="font-size:16px;font-weight:600;color:#7c2d12;">How would you rate the severity?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_temp, col_hr = st.columns(2)
    with col_temp:
        temperature_level = st.selectbox("Temperature", options=list(temp_map.keys()), index=1)
    with col_hr:
        heart_rate_level  = st.selectbox("Heart rate", options=list(hr_map.keys()), index=1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 4 — Medical History
    st.markdown("""
    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="background:#059669;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">4</span>
            <span style="font-size:16px;font-weight:600;color:#064e3b;">Do you have any of the following?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ch1, ch2, ch3, _ = st.columns(4)
    with ch1: hypertension  = st.checkbox("🩺 High Blood Pressure")
    with ch2: heart_disease = st.checkbox("❤️ Heart Disease")
    with ch3: asthma        = st.checkbox("💨 Asthma")

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 5 — Patient Info
    st.markdown("""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;
                padding:20px 24px;margin-bottom:24px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#475569;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">5</span>
            <span style="font-size:16px;font-weight:600;color:#1e293b;">Patient Information</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_age, col_gen = st.columns(2)
    with col_age:
        age    = st.number_input("Age", min_value=1, max_value=120, value=35)
    with col_gen:
        gender = st.selectbox("Gender", options=['Female', 'Male'])

    submitted = st.form_submit_button("Get AI Recommendation →")

# ── Result ────────────────────────────────────────────────────────────────────
if submitted:
    patient = pd.DataFrame([{
        'age'              : age,
        'gender'           : gender_map.get(gender, 0),
        'fever'            : int(fever),
        'cough'            : int(cough),
        'headache'         : int(headache),
        'chest_pain'       : int(chest_pain),
        'stomach_pain'     : int(stomach_pain),
        'shortness_breath' : int(shortness_breath),
        'nausea_vomiting'  : int(nausea_vomiting),
        'dizziness'        : int(dizziness),
        'skin_rash'        : int(skin_rash),
        'temperature_level': temp_map.get(temperature_level, 1),
        'heart_rate_level' : hr_map.get(heart_rate_level, 1),
        'duration'         : dur_map.get(duration, 1),
        'asthma'           : int(asthma),
        'hypertension'     : int(hypertension),
        'heart_disease'    : int(heart_disease),
        'chief_complaint'  : cc_map.get(chief_complaint, 9)
    }])

    patient_scaled = patient.copy()
    patient_scaled[cols_to_scale] = scaler.transform(patient[cols_to_scale])

    pred       = model.predict(patient_scaled[features])[0]
    proba      = model.predict_proba(patient_scaled[features])[0]
    dept_name  = dept_map_inv[pred]
    confidence = proba[pred] * 100
    info       = DEPT_INFO[dept_name]

    st.markdown("---")
    st.markdown("""
    <div style="font-size:22px;font-weight:700;color:#111827;margin-bottom:4px;">AI Recommendation</div>
    <div style="font-size:14px;color:#6b7280;margin-bottom:1.5rem;">Based on the information you provided</div>
    """, unsafe_allow_html=True)

    res_col, prob_col = st.columns([3, 2])

    with res_col:
        steps_html = ''.join(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="color:{info["color"]};font-size:14px;">📍</span>'
            f'<span style="font-size:14px;color:#374151;">{step}</span></div>'
            for step in info['next']
        )
        st.markdown(f"""
        <div style="background:{info['bg']};border:1.5px solid {info['border']};
                    border-radius:16px;padding:28px 32px;">
            <div style="font-size:44px;margin-bottom:12px;">{info['icon']}</div>
            <div style="font-size:26px;font-weight:700;color:{info['color']};margin-bottom:8px;">{dept_name}</div>
            <div style="font-size:14px;color:#374151;margin-bottom:20px;">
                Our AI suggests you visit the <strong>{dept_name}</strong> Department.
            </div>
            <div style="font-size:11px;font-weight:600;color:{info['color']};text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:8px;">Why?</div>
            <div style="font-size:14px;color:#4b5563;margin-bottom:20px;">{info['desc']} Your reported symptoms and vitals match patients typically directed to this department.</div>
            <div style="font-size:11px;font-weight:600;color:{info['color']};text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:10px;">What to do next?</div>
            {steps_html}
            <div style="margin-top:20px;padding:12px 16px;background:rgba(0,0,0,0.05);
                        border-radius:10px;font-size:12px;color:#6b7280;line-height:1.5;">
                ⚠️ This is an AI suggestion, not a medical diagnosis. Please consult a doctor for further evaluation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with prob_col:
        st.markdown(f"""
        <div style="background:white;border:1px solid #e5e7eb;border-radius:16px;padding:24px;">
            <div style="font-size:14px;font-weight:600;color:#111827;margin-bottom:16px;">
                Confidence by department
            </div>
        """, unsafe_allow_html=True)

        sorted_depts = sorted(dept_map_inv.items(), key=lambda x: proba[x[0]], reverse=True)
        bars_html = ""
        for idx, dname in sorted_depts:
            pct    = proba[idx] * 100
            dinfo  = DEPT_INFO[dname]
            is_top = dname == dept_name
            bars_html += f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                    <span style="font-size:13px;font-weight:{'700' if is_top else '400'};
                                 color:{'#111827' if is_top else '#6b7280'};">
                        {dinfo['icon']} {dname}
                    </span>
                    <span style="font-size:13px;font-weight:{'700' if is_top else '400'};
                                 color:{dinfo['color'] if is_top else '#9ca3af'};">
                        {pct:.1f}%
					</span>
                </div>
                <div style="background:#f3f4f6;border-radius:6px;height:8px;overflow:hidden;">
                    <div style="background:{'linear-gradient(90deg,'+dinfo['color']+','+dinfo['border']+')' if is_top else '#e5e7eb'};
                                height:100%;border-radius:6px;width:{pct}%;
                                transition:width 0.5s ease;"></div>
                </div>
            </div>"""

        st.markdown(bars_html + """
            <div style="margin-top:20px;background:#eff6ff;border:1px solid #bfdbfe;
                        border-radius:10px;padding:12px 14px;font-size:12px;color:#1e40af;">
                <strong>Model:</strong> KNN (k=7) · 102,000 patients · 99.5% accuracy<br>
                <strong>Powered by:</strong> Future Classroom ML
            </div>
        </div>
        """, unsafe_allow_html=True)
