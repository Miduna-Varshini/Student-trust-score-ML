import streamlit as st
import joblib
import time
import random
import pandas as pd
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student Trust Score System",
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

# ================= UI STYLE =================
st.markdown("""
<style>
body {
    background-color: #f8fafc;
}
.header {
    background: linear-gradient(90deg, #0f172a, #1e3a8a);
    padding: 25px;
    border-radius: 12px;
    color: white;
    text-align: center;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.footer {
    background-color:#0f172a;
    padding:12px;
    color:white;
    text-align:center;
    border-radius:10px;
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="header">
<h1>🎓 Student Trust Score Monitoring System</h1>
<p>ML-based Behavioral Analysis & Anomaly Detection</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ================= LOGIN SECTION =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔐 Student Login")

student_id = st.text_input("Student ID")
student_name = st.text_input("Student Name")

login_btn = st.button("Login")
st.markdown("</div>", unsafe_allow_html=True)

# ================= LOGIN ACTION =================
if login_btn and student_id and student_name:
    st.session_state.start_time = time.time()
    st.success(f"Login detected for {student_name}")

# ================= MONITORING =================
if st.session_state.start_time:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📡 Live Behavior Monitoring")

    session_duration = round(time.time() - st.session_state.start_time, 2)
    login_hour = datetime.now().hour

    # -------- AUTO-GENERATED FEATURES (6 FEATURES) --------
    click_speed = random.randint(20, 80)
    device_change = random.choice([0, 1])
    login_frequency = random.randint(1, 5)
    typing_speed = random.randint(30, 90)

    features = [[
        login_hour,
        session_duration,
        click_speed,
        device_change,
        login_frequency,
        typing_speed
    ]]

    # -------- ML PREDICTION --------
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    trust_score = max(0, min(100, 100 - abs(prediction) * random.randint(15, 25)))
    status = "✅ Trusted" if trust_score >= 60 else "⚠ Suspicious"

    col1, col2, col3 = st.columns(3)
    col1.metric("Session Duration (sec)", session_duration)
    col2.metric("Click Speed", click_speed)
    col3.metric("Trust Score", trust_score)

    st.info(f"Status: {status}")

    # -------- SAVE SESSION --------
    if st.button("End Session & Save"):
        entry = {
            "Student ID": student_id,
            "Name": student_name,
            "Login Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Session Duration (sec)": session_duration,
            "Trust Score": trust_score,
            "Status": status
        }
        st.session_state.report.append(entry)
        st.session_state.start_time = None
        st.success("Session saved successfully!")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= REPORT =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📊 Student Entry Report")

if st.session_state.report:
    df = pd.DataFrame(st.session_state.report)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No student sessions recorded yet.")

st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
<p>© 2025 Student Trust Score System | ML Internship Project</p>
</div>
""", unsafe_allow_html=True)
