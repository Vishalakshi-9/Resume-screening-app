import streamlit as st
import pandas as pd
from utils import extract_text_from_pdf, clean_text, rank_resumes, detect_job_role
from roles import job_roles

st.set_page_config(page_title="Resume Screening App", layout="wide")

st.title("AI-Powered Resume Screening System")

# Section 2: Predict Role
# ----------------------------
st.header("Predict Suitable Job Role from Resume")

single_file = st.file_uploader("Upload one resume to predict job role", type=["pdf"], key="role_predict")
if single_file is not None:
    raw_text = extract_text_from_pdf(single_file)
    cleaned = clean_text(raw_text)
    role, score = detect_job_role(cleaned, job_roles)
    st.success(f" Best Matched Role: **{role}** (Score: `{score:.3f}`)")
    with st.expander("Resume Preview"):
        st.write(raw_text[:1000] + " ...")

