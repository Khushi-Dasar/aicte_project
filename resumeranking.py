import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------- Function to extract text from PDF --------
def extract_text_from_pdf(file):
    text = ""
    try:
        pdf = PdfReader(file)
        for page in pdf.pages:
            content = page.extract_text()
            if content:  # Avoid None values
                text += content
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
    return text


# -------- Function to rank resumes --------
def rank_resumes(job_description, resumes):
    # Combine job description and resumes
    documents = [job_description] + resumes

    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(documents).toarray()

    # Compute cosine similarity
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    similarities = cosine_similarity([job_vector], resume_vectors).flatten()

    return similarities


# -------- Streamlit App --------
st.title("📄 AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

# Process resumes
if uploaded_files and job_description:
    st.header("📊 Ranking Resumes")

    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Create results table
    results = pd.DataFrame({
        "Resume": [file.name for file in uploaded_files],
        "Score": scores
    })

    # Sort results
    results = results.sort_values(by="Score", ascending=False)

    # Display results
    st.write(results)
