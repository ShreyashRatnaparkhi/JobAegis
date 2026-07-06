# ==========================================================
# JobAegis - Fake Job Detection System
# Author: Shreyash
# ==========================================================

import streamlit as st
import joblib
import re
import numpy as np

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------

st.set_page_config(
    page_title="JobAegis",
    page_icon="🛡️",
    layout="wide"
)

# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------

@st.cache_resource
def load_files():

    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/tfidf.pkl")

    return model, vectorizer

model, vectorizer = load_files()

# ----------------------------------------------------------
# TEXT CLEANING
# ----------------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

with st.sidebar:

    st.title("🛡️ JobAegis")

    st.markdown("---")

    st.subheader("About")

    st.write(
        """
JobAegis is an AI-powered Fake Job Posting Detection System.

It analyzes job advertisements using Machine Learning and Natural Language Processing.

Developed using:

• Python

• Streamlit

• Scikit-Learn

• XGBoost

• TF-IDF
"""
    )

    st.markdown("---")

    st.success("Developed by Shreyash")

# ----------------------------------------------------------
# MAIN PAGE
# ----------------------------------------------------------

st.title("🛡️ JobAegis")

st.subheader("Fake Job Posting Detection using Machine Learning")

st.write(
"""
Paste a complete job posting below and click **Analyze Job Posting**.
"""
)

job_text = st.text_area(

    "Job Description",

    height=320,

    placeholder="Paste the complete job description here..."
)





# ----------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------

if st.button("🔍 Analyze Job Posting", use_container_width=True):

    if job_text.strip() == "":

        st.warning("⚠ Please enter a job description.")

    else:

        with st.spinner("Analyzing job posting..."):

            cleaned_text = clean_text(job_text)

            vector = vectorizer.transform([cleaned_text])

            prediction = model.predict(vector)[0]

        st.markdown("---")

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        if prediction == 1:

            st.error("🚨 FAKE JOB POSTING DETECTED")

        else:

            st.success("✅ GENUINE JOB POSTING")

        # --------------------------------------------------
        # CONFIDENCE SCORE
        # --------------------------------------------------

        st.subheader("Prediction Confidence")

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(vector)[0]

            genuine_prob = float(probabilities[0])
            fake_prob = float(probabilities[1])

            col1, col2 = st.columns(2)

            with col1:

                st.write("### Genuine Probability")

                st.progress(genuine_prob)

                st.write(f"{genuine_prob*100:.2f}%")

            with col2:

                st.write("### Fake Probability")

                st.progress(fake_prob)

                st.write(f"{fake_prob*100:.2f}%")

        else:

            st.info(
                "Confidence scores are not available for this model."
            )

        # --------------------------------------------------
        # ANALYSIS SUMMARY
        # --------------------------------------------------

        st.markdown("---")

        st.subheader("Analysis Summary")

        if prediction == 1:

            st.write(
                """
Possible reasons:

- Unrealistic salary
- Suspicious wording
- Missing company details
- Excessive promises
- Scam-like language

Always verify jobs from official company websites before applying.
"""
            )

        else:

            st.write(
                """
This posting appears to be legitimate based on the trained machine learning model.

Always verify:

- Company website
- Official email domain
- LinkedIn profile
- Interview process

before accepting any offer.
"""
            )

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.markdown("---")

st.caption(
    "🛡️ JobAegis | Fake Job Detection using Machine Learning | Developed by Shreyash"
)