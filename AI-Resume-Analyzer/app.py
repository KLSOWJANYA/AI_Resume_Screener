import streamlit as st
import pickle
import numpy as np
from PyPDF2 import PdfReader

# ----------- Load Model and Vectorizer -----------
with open('resume_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('skills_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# ----------- Dummy Extraction Functions -----------
def extract_skills(text):
    # Dummy extraction of skills from text using a predefined list
    skills_keywords = ['python', 'java', 'sql', 'machine learning', 'deep learning', 'flask', 'streamlit', 'pandas']
    detected_skills = [kw for kw in skills_keywords if kw in text.lower()]
    return ", ".join(detected_skills)

def extract_education(text):
    # Dummy extraction: for demo purposes, we assume Bachelor's is found
    return "Bachelor's"

def extract_certifications(text):
    # Dummy extraction: assume no certifications unless the word 'certification' is found
    return "None" if "certification" not in text.lower() else "Some"

def extract_projects_count(text):
    # Dummy extraction: count the occurrences of the word 'project'
    count = text.lower().count("project")
    return str(count)

def extract_job_role(text):
    # Dummy extraction: assume a default job role
    return "Data Scientist"

# ----------- Extract Resume Text -----------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# ----------- Feature Extraction with Detailed Feedback -----------
def extract_features_from_text(text):
    # Extract information using dummy functions
    skills_text = extract_skills(text)
    education = extract_education(text)
    certifications = extract_certifications(text)
    projects = extract_projects_count(text)
    job_role = extract_job_role(text)

    # Transform the detected skills using the fitted vectorizer
    skills_vec = vectorizer.transform([skills_text]).toarray()

    # Convert other features to dummy numeric values:
    # For education, we'll assume Bachelor's (2) is good; lower is 1.
    edu_val = 2 if "bachelor" in education.lower() else 1
    # If no certifications, we mark it as 0.
    cert_val = 0 if certifications.lower() in ["none", ""] else 1
    # For projects, if count is 0 then value is 0; else, positive.
    try:
        proj_val = int(projects)
    except:
        proj_val = 0
    # Job role is assumed to be a dummy constant (1) for now.
    role_val = 1

    # Combine all features
    others = np.array([[edu_val, cert_val, proj_val, role_val]])
    features = np.concatenate([skills_vec, others], axis=1)

    # Build reasons for not being selected based on our dummy criteria
    reasons = []
    if len(skills_text.split(',')) < 3:
        reasons.append("Insufficient relevant technical skills detected.")
    if cert_val == 0:
        reasons.append("No certifications provided.")
    if proj_val == 0:
        reasons.append("No project experience mentioned.")
    if edu_val < 2:
        reasons.append("Educational qualifications appear to be low.")

    # If no reason is generated, provide a default message.
    if not reasons:
        reasons.append("Your resume may be lacking in qualitative details not captured by our current metrics.")

    return features, reasons, skills_text

# ----------- Streamlit UI -----------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer")
st.write("Upload your resume PDF to check your hiring likelihood prediction and receive feedback.")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    if resume_text:
        features, reasons, detected_skills = extract_features_from_text(resume_text)
        prediction = model.predict(features)[0]

        st.markdown("### Hiring Likelihood Prediction:")
        if prediction == 1:
            st.success("Selected")
        else:
            st.error("Not Selected")
            st.markdown("### Reasons for Not Being Selected:")
            for reason in reasons:
                st.write(f"- {reason}")

        st.markdown("### Detected Skills:")
        st.write(detected_skills if detected_skills else "No skills detected.")
    else:
        st.error("Unable to extract text from the uploaded file. Please try another PDF.")
