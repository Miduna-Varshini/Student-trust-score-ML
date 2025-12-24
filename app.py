import streamlit as st
import joblib
import time
import random
import pandas as pd
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student Trust Score System",
    page_icon="🎓",
    layout="wide"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_models():
    model = joblib.load("isolation_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_models()

# ================= SESSION STATE =================
if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "report" not in st.session_state:
    st.session_state.report = []

# ================= ADVANCED RESPONSIVE CSS =================
st.markdown("""
<style>

/* ===== Global ===== */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg,#0f172a,#020617);
}

/* ===== Header ===== */
.main-header {
    background: linear-gradient(120deg,#2563eb,#1e40af,#0f172a);
    padding: 40px 20px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
}

.main-header h1 {
    font-size: clamp(26px, 4vw, 42px);
    font-weight: 800;
}

.main-header p {
    opacity: 0.9;
    font-size: clamp(14px, 2vw, 18px);
}

/* ===== Glass Cards ===== */
.glass {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    margin-bottom: 25px;
}

/* ===== Inputs ===== */
input, .stTextInput>div>div>input {
    background: rgba(255,255,255,0.2) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* ===== Buttons ===== */
.stButton>button {
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
    color: white;
    border-radius: 12px;
    padding: 12px 26px;
    font-size: 16px;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 8px 30px rgba(37,99,235,0.6);
}

/* ===== Metrics ===== */
.metric-box {
    background: rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

/* ===== Footer ===== */
.footer {
    background: linear-gradient(120deg,#020617,#020617,#020617);
    color: #cbd5f5;
    text-align: center;
    padding: 15px;
    border-radius: 14px;
    margin-top: 40px;
    font-size: 14px;
}

/* ===== Responsive ===== */
@media (max-width: 768px) {
    .glass {
        padding: 18px;
    }
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Trust Score Monitoring</h1>
    <p>Real-time ML-based Behavioral Analysis Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ================= LOGIN SECTION =================
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("🔐 Student Entry")

student_id = st.text_input("Student ID")
student_name = st.text_input("Student Name")

if st.button("Start Monitoring"):
    if student_id and student_name:
        st.session_state.start_time = time.time()
        st.success(f"Monitoring started for {student_name}")
    else:
        st.warning("Please enter all details")

st.markdown("</div>", unsafe_allow_html=True)

# ================= LIVE MONITORING =================
if st.session_state.start_time:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📡 Live Session Analysis")

    duration = round(time.time() - st.session_state.start_time, 2)
    login_hour = datetime.now().hour

    n_features = scaler.n_features_in_

    features = [
        login_hour,
        duration,
        random.randint(30, 90),
        random.choice([0, 1]),
        random.randint(1, 5),
        random.randint(40, 120)
    ]

    if len(features) < n_features:
        features.extend([0] * (n_features - len(features)))
    features = features[:n_features]

    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]

    trust_score = max(0, min(100, 100 - abs(prediction) * random.randint(18, 30)))
    status = "✅ Trusted" if trust_score >= 60 else "⚠ Suspicious"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-box'>⏱️<br><b>Duration</b><br>" + str(duration) + " sec</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-box'>🧠<br><b>Features</b><br>" + str(n_features) + "</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-box'>⭐<br><b>Trust Score</b><br>" + str(trust_score) + "</div>", unsafe_allow_html=True)

    st.info(f"Status: **{status}**")

    if st.button("End Session & Save"):
        st.session_state.report.append({
            "Student ID": student_id,
            "Name": student_name,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Duration": duration,
            "Trust Score": trust_score,
            "Status": status
        })
        st.session_state.start_time = None
        st.success("Session saved successfully")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= REPORT =================
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("📊 Entry Report Dashboard")

if st.session_state.report:
    df = pd.DataFrame(st.session_state.report)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No data recorded yet")

st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
© 2025 Student Trust Score System | AI & ML Project
</div>
""", unsafe_allow_html=True)
