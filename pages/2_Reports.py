import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("📊 Student Entry Reports")

if "report" in st.session_state and st.session_state.report:
    df = pd.DataFrame(st.session_state.report)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇ Download Report",
        df.to_csv(index=False),
        "student_trust_report.csv"
    )
else:
    st.warning("No student data available yet")
