import streamlit as st
import joblib
import time
import random
from datetime import datetime

st.set_page_config(layout="wide")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_models():
    model = joblib.load("isolation_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_models()

# ---------- SESSION ----------
if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "report" not in st.session_state:
    st.session_state.report = []

# ---------- CSS ----------
st.markdown("""
<style>
.card {
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0 8px 20px rgba(0,0,0,0.1);
}
.safe {color:#16a34a;font-weight:700;}
.risk {color:#dc2626;font-weight:700;}
</style>
""", unsafe_allow_html=True)

st.title("📡 Live Student Monitoring")

# ---------- LOGIN ----------
st.markdown("<div class='card'>", unsafe_allow_html=True)
student_id = st.text_input("Student ID")
student_name = st.text_input("Student Name")
login = st.button("🚀 Start Monitoring")
st.markdown("</div>", unsafe_allow_html=True)

if login and student_id and student_name:
    st.session_state.start_time = time.time()
    st.success(f"Monitoring started for {student_name}")

# ---------- MONITOR ----------
if st.session_state.start_time:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    duration = round(time.time() - st.session_state.start_time, 2)
    hour = datetime.now().hour
    n = scaler.n_features_in_

    features = [
        hour,
        duration,
        random.randint(20, 80),
        random.choice([0, 1]),
        random.randint(1, 5),
        random.randint(30, 90)
    ]

    features = features[:n] + [0]*(n-len(features))
    scaled = scaler.transform([features])
    pred = model.predict(scaled)[0]

    trust = max(0, min(100, 100 - abs(pred)*random.randint(15,25)))
    status = "Trusted" if trust >= 60 else "Suspicious"

    col1, col2, col3 = st.columns(3)
    col1.metric("Session Time (sec)", duration)
    col2.metric("Trust Score", trust)
    col3.metric("Status", status)

    if st.button("End & Save"):
        st.session_state.report.append({
            "ID": student_id,
            "Name": student_name,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Duration": duration,
            "Trust Score": trust,
            "Status": status
        })
        st.session_state.start_time = None
        st.success("Session saved!")

    st.markdown("</div>", unsafe_allow_html=True)
