import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Resume Screening System",
    layout="centered"
)

st.title("AI Resume Screening System")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    resume_df = pd.read_csv("data/Resume.csv")
    job_df = pd.read_csv("data/DataScientist.csv")
    return resume_df, job_df


resume_df, job_df = load_data()

# -----------------------------
# SAFE COLUMN HANDLING
# -----------------------------
def get_resume_text_column(df):
    if "clean_resume" in df.columns:
        return "clean_resume"
    elif "Resume_str" in df.columns:
        return "Resume_str"
    else:
        raise ValueError("No valid resume text column found")


def get_job_text_column(df):
    if "clean_jd" in df.columns:
        return "clean_jd"
    elif "Job Description" in df.columns:
        return "Job Description"
    else:
        raise ValueError("No valid job description column found")


resume_text_col = get_resume_text_column(resume_df)
job_text_col = get_job_text_column(job_df)

# -----------------------------
# TEXT VECTORIZATION
# -----------------------------
@st.cache_data
def create_vectors(resume_df, job_df, resume_col, job_col):
    combined_text = (
        resume_df[resume_col].astype(str).fillna("").tolist()
        + job_df[job_col].astype(str).fillna("").tolist()
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    resume_vectors = tfidf_matrix[:len(resume_df)]
    job_vectors = tfidf_matrix[len(resume_df):]

    return resume_vectors, job_vectors


resume_vectors, job_vectors = create_vectors(
    resume_df, job_df, resume_text_col, job_text_col
)

# -----------------------------
# USER INPUT
# -----------------------------
selected_category = st.selectbox(
    "Select Resume Category",
    sorted(resume_df["Category"].dropna().unique())
)

# -----------------------------
# FILTER RESUMES
# -----------------------------
filtered_resumes = resume_df[resume_df["Category"] == selected_category]

if filtered_resumes.empty:
    st.warning("No resumes found for this category.")
    st.stop()

resume_index = filtered_resumes.index[0]
resume_vector = resume_vectors[resume_index]

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
similarity_scores = cosine_similarity(resume_vector, job_vectors).flatten()

job_df["Match Score (%)"] = similarity_scores * 100

top_jobs = (
    job_df.sort_values(by="Match Score (%)", ascending=False)
    .head(5)[
        [
            "Job Title",
            "Company Name",
            "Location",
            "Match Score (%)"
        ]
    ]
)

# -----------------------------
# OUTPUT
# -----------------------------
st.subheader("Top Job Matches")

st.dataframe(
    top_jobs.style.format({"Match Score (%)": "{:.2f}"})
)
