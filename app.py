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
# TEXT VECTORIZATION
# -----------------------------
@st.cache_data
def create_vectors(resume_df, job_df):
    combined_text = (
        resume_df["clean_resume"].astype(str).tolist()
        + job_df["clean_jd"].astype(str).tolist()
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    resume_vectors = tfidf_matrix[:len(resume_df)]
    job_vectors = tfidf_matrix[len(resume_df):]

    return resume_vectors, job_vectors


resume_vectors, job_vectors = create_vectors(resume_df, job_df)

# -----------------------------
# USER INPUT
# -----------------------------
selected_category = st.selectbox(
    "Select Resume Category",
    sorted(resume_df["Category"].unique())
)

# -----------------------------
# FILTER RESUME
# -----------------------------
filtered_resumes = resume_df[resume_df["Category"] == selected_category]

if filtered_resumes.empty:
    st.warning("No resumes found for this category.")
else:
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
