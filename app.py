import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Intelligent Student Identity Trust System",
    layout="wide"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_models():
    model = joblib.load("isolation_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_models()

# ================= UI HEADER =================
st.markdown(
    """
    <div style="background:linear-gradient(to right,#4facfe,#00f2fe);
                padding:20px;border-radius:10px">
        <h1 style="color:white;text-align:center;">🎓 Intelligent Student Identity Trust System</h1>
        <p style="color:white;text-align:center;">Real-time student behavior & room entry monitoring</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ================= SIDEBAR INPUT =================
st.sidebar.header("🧑‍🎓 Student Activity Input")

student_name = st.sidebar.text_input("Student Name")

login_hour = st.sidebar.slider("Login Hour (0-23)", 0, 23, 10)
day_type = st.sidebar.selectbox("Day Type", ["Weekday", "Weekend"])
session_duration = st.sidebar.slider("Session Duration (minutes)", 1, 300, 30)
actions_per_minute = st.sidebar.slider("Actions per Minute", 1, 100, 20)
time_between_actions = st.sidebar.slider("Time Between Actions (seconds)", 0.1, 10.0, 3.0)
files_accessed = st.sidebar.slider("Files Accessed", 0, 100, 10)
device_change = st.sidebar.selectbox("Device Change Detected?", ["No", "Yes"])

# ================= PREPARE INPUT =================
day_encoded = 0 if day_type == "Weekday" else 1
device_encoded = 1 if device_change == "Yes" else 0

input_data = [[
    login_hour,
    day_encoded,
    session_duration,
    actions_per_minute,
    time_between_actions,
    files_accessed,
    device_encoded
]]

scaled_input = scaler.transform(input_data)

# ================= PREDICTION =================
if st.sidebar.button("🔍 Predict Trust Score"):

    if student_name.strip() == "":
        st.error("⚠️ Please enter Student Name")
    else:
        prediction = model.predict(scaled_input)[0]
        score = int(100 - abs(model.decision_function(scaled_input)[0] * 50))

        if prediction == -1:
            status = "Anomalous"
            risk = "High"
        else:
            status = "Normal"
            risk = "Low"

        # ================= DISPLAY RESULT =================
        col1, col2, col3 = st.columns(3)

        col1.metric("Trust Score", f"{score}/100")
        col2.metric("Behavior Status", status)
        col3.metric("Risk Level", risk)

        # ================= SAVE ENTRY TO CSV =================
        log_file = "entry_log.csv"

        log_data = {
            "Student Name": student_name,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Entry Time": datetime.now().strftime("%H:%M:%S"),
            "Trust Score": score,
            "Behavior Status": status,
            "Risk Level": risk,
            "Device Changed": device_change
        }

        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            df = pd.concat([df, pd.DataFrame([log_data])], ignore_index=True)
        else:
            df = pd.DataFrame([log_data])

        df.to_csv(log_file, index=False)

        st.success("✅ Student entry logged successfully")

# ================= REPORT SECTION =================
st.markdown("---")
st.subheader("📋 Room Entry Report")

if os.path.exists("entry_log.csv"):
    report_df = pd.read_csv("entry_log.csv")

    st.dataframe(report_df, use_container_width=True)

    # SUMMARY
    st.markdown("### 📊 Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Entries", len(report_df))
    c2.metric("Normal", len(report_df[report_df["Behavior Status"] == "Normal"]))
    c3.metric("Anomalous", len(report_df[report_df["Behavior Status"] == "Anomalous"]))

    # DOWNLOAD
    st.download_button(
        label="⬇️ Download Report (CSV)",
        data=report_df.to_csv(index=False),
        file_name="student_entry_report.csv",
        mime="text/csv"
    )
else:
    st.info("No entries recorded yet.")
