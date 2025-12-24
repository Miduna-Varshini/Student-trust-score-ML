import streamlit as st
import joblib
import time
import random
import pandas as pd
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student Trust Score Portal",
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

# ================= WEBSITE CSS =================
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body {
    background-color: #020617;
    font-family: 'Segoe UI', sans-serif;
}

/* ---------- NAVBAR ---------- */
.navbar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: linear-gradient(90deg,#020617,#020617,#020617);
    padding: 16px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1e293b;
}

.navbar h2 {
    color: #38bdf8;
    margin: 0;
}

.navbar span {
    color: #cbd5f5;
    margin-left: 25px;
    font-size: 16px;
}

/* ---------- HERO ---------- */
.hero {
    padding: 80px 20px;
    text-align: center;
    background: radial-gradient(circle at top,#1e3a8a,#020617);
}

.hero h1 {
    font-size: clamp(32px,5vw,56px);
    color: white;
    font-weight: 800;
}

.hero p {
    color: #cbd5f5;
    max-width: 750px;
    margin: auto;
    font-size: 18px;
}

/* ---------- SECTION ---------- */
.section {
    padding: 60px 40px;
}

/* ---------- CARD ---------- */
.card {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 30px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.4);
    backdrop-filter: blur(15px);
}

/* ---------- BUTTON ---------- */
.stButton>button {
    background: linear-gradient(135deg,#38bdf8,#2563eb);
    border-radius: 12px;
    padding: 14px 30px;
    font-size: 16px;
    color: white;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.05);
}

/* ---------- METRIC ---------- */
.metric {
    background: rgba(255,255,255,0.12);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
}

/* ---------- FOOTER ---------- */
.footer {
    background: #020617;
    padding: 25px;
    text-align: center;
    color: #94a3b8;
    border-top: 1px solid #1e293b;
}

/* ---------- MOBILE ---------- */
@media(max-width:768px){
    .section {
        padding: 40px 20px;
    }
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR =================
st.markdown("""
<div class="navbar">
    <h2>🎓 TrustScore</h2>
    <div>
        <span>Home</span>
        <span>Live Monitor</span>
        <span>Report</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div class="hero">
    <h1>Student Trust Score Monitoring Portal</h1>
    <p>
        A real-time AI-powered system that automatically analyzes student behavior,
        session activity, and anomaly patterns to generate trust scores.
    </p>
</div>
""", unsafe_allow_html=True)

# ================= LOGIN SECTION =================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("🔐 Student Entry")
student_id = st.text_input("Student ID")
student_name = st.text_input("Student Name")

if st.button("Start Session"):
    if student_id and student_name:
        st.session_state.start_time = time.time()
        st.success(f"Session started for {student_name}")
    else:
        st.warning("Enter all details")

st.markdown("</div></div>", unsafe_allow_html=True)

# ================= LIVE MONITOR =================
if st.session_state.start_time:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📡 Live Monitoring")

    duration = round(time.time() - st.session_state.start_time, 2)
    hour = datetime.now().hour

    n_features = scaler.n_features_in_
    features = [
        hour,
        duration,
        random.randint(20,100),
        random.choice([0,1]),
        random.randint(1,5),
        random.randint(40,120)
    ]

    if len(features) < n_features:
        features.extend([0]*(n_features-len(features)))
    features = features[:n_features]

    scaled = scaler.transform([features])
    pred = model.predict(scaled)[0]

    trust_score = max(0, min(100, 100 - abs(pred)*25))
    status = "✅ Trusted" if trust_score >= 60 else "⚠ Suspicious"

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric'>⏱️<br><b>{duration}s</b></div>",unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric'>🧠<br><b>{n_features} Features</b></div>",unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric'>⭐<br><b>{trust_score}</b></div>",unsafe_allow_html=True)

    st.info(f"Status: **{status}**")

    if st.button("End Session"):
        st.session_state.report.append({
            "Student ID": student_id,
            "Name": student_name,
            "Duration": duration,
            "Trust Score": trust_score,
            "Status": status
        })
        st.session_state.start_time = None
        st.success("Session saved")

    st.markdown("</div></div>", unsafe_allow_html=True)

# ================= REPORT =================
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📊 Student Report")

if st.session_state.report:
    df = pd.DataFrame(st.session_state.report)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No data available")

st.markdown("</div></div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
© 2025 TrustScore Platform • AI Monitoring System
</div>
""", unsafe_allow_html=True)
