import streamlit as st
import pickle

# Load model and vectorizer
with open("job_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("🔍 Fake Job Posting Detector")

user_input = st.text_area("Paste the job description here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Basic clean (use full preprocessing if needed)
        user_input_clean = user_input.lower()

        # Vectorize input
        input_vector = vectorizer.transform([user_input_clean])

        # Predict
        result = model.predict(input_vector)[0]

        if result == 1:
            st.error("❌ This job posting is likely **FAKE**.")
        else:
            st.success("✅ This job posting is likely **REAL**.")
