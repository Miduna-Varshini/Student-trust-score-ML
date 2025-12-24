import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student Trust Scoring System",
    page_icon="🎓",
    layout="wide"
)
st.markdown(
    """
    <style>
        background-color: #15e6df;
    .footer {
        background-color: #306998;
        padding: 15px;
        color: white;
        text-align: center;
        border-radius: 10px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🎓 Intelligent Student Identity Trust System")
st.markdown(
    "This application uses a **Machine Learning model (.pkl)** to evaluate "
    "student behavior and generate a **trust score** in real time."
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_models():
    # Since files are in the same folder as app.py
    model = joblib.load("isolation_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_models()


# ================= LOAD DATA =================
df = pd.read_excel("student_activity.xlsx")
# ================= SIDEBAR INPUT =================
st.sidebar.header("📥 Student Activity Input")

login_hour = st.sidebar.slider("Login Hour (0–23)", 0, 23, 9)
day_type = st.sidebar.selectbox("Day Type", ["Weekday", "Weekend"])
session_duration = st.sidebar.slider("Session Duration (minutes)", 5, 300, 60)
actions_per_minute = st.sidebar.slider("Actions per Minute", 1, 100, 15)
time_between_actions = st.sidebar.slider("Time Between Actions (seconds)", 0.1, 20.0, 4.0)
files_accessed = st.sidebar.slider("Files Accessed", 0, 50, 5)
device_change = st.sidebar.selectbox("Device Change Detected?", ["No", "Yes"])

# Convert categorical values
day_type = 0 if day_type == "Weekday" else 1
device_change = 0 if device_change == "No" else 1

# ================= PREDICTION =================
if st.sidebar.button("🔍 Evaluate Trust"):

    input_data = pd.DataFrame([{
        "login_hour": login_hour,
        "day_type": day_type,
        "session_duration": session_duration,
        "actions_per_minute": actions_per_minute,
        "time_between_actions": time_between_actions,
        "files_accessed": files_accessed,
        "device_change": device_change
    }])

    # Scale input
    scaled_input = scaler.transform(input_data)

    # Model prediction
    anomaly_label = model.predict(scaled_input)[0]
    anomaly_score = model.decision_function(scaled_input)[0]

    # Trust score calculation
    trust_score = int(np.clip((anomaly_score + 0.5) * 100, 0, 100))

    # Risk level
    if trust_score >= 80:
        risk_level = "🟢 High Trust"
    elif trust_score >= 50:
        risk_level = "🟡 Medium Trust"
    else:
        risk_level = "🔴 Low Trust"

    # ================= OUTPUT =================
    st.subheader("📊 Trust Assessment Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Trust Score", f"{trust_score}/100")
    col2.metric("Behavior Status", "Normal" if anomaly_label == 1 else "Anomalous")
    col3.metric("Risk Level", risk_level)

    # ================= EXPLANATION =================
    st.subheader("🧠 Explanation")

    explanation = []

    if login_hour < 6 or login_hour > 22:
        explanation.append("Login time is outside normal study hours.")
    if actions_per_minute > 40:
        explanation.append("Unusually high activity speed detected.")
    if device_change == 1:
        explanation.append("Student accessed the system from a new device.")
    if session_duration > 180:
        explanation.append("Very long session duration observed.")

    if explanation:
        for e in explanation:
            st.write("•", e)
    else:
        st.write("Student behavior appears normal and consistent.")

# ================= FOOTER =================
st.markdown("---")
st.caption("Developed using Machine Learning & Streamlit | Academic Project")
