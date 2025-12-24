import streamlit as st

st.set_page_config(
    page_title="Student Trust Score System",
    layout="wide"
)

# ---------- CSS ----------
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg,#0f172a,#1e40af);
    padding: 50px;
    border-radius: 20px;
    color: white;
    text-align: center;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="hero">
<h1>🎓 Student Trust Score System</h1>
<p>AI-based Behavioral Monitoring & Cyber-Aware Analytics</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------- INFO ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'><h3>📡 Live Monitoring</h3><p>Real-time student behavior analysis</p></div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'><h3>🔐 ML Security</h3><p>Anomaly detection using Isolation Forest</p></div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'><h3>📊 Reports</h3><p>Entry & trust score reports</p></div>", unsafe_allow_html=True)

st.success("👈 Use the left sidebar to navigate like a real website")
