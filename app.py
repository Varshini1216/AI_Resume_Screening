import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Screening System")

resume_df = pd.read_csv("data/Resume.csv")
job_df = pd.read_csv("data/DataScientist.csv")

resume_text = st.selectbox("Select Resume", resume_df['Category'].unique())

resume_idx = resume_df[resume_df['Category'] == resume_text].index[0]

# Load vectors (you will save them next)
with open("vectors.pkl", "rb") as f:
    resume_vectors, job_vectors = pickle.load(f)

scores = cosine_similarity(
    resume_vectors[resume_idx].reshape(1, -1),
    job_vectors
)[0]

top_jobs = scores.argsort()[-5:][::-1]

st.dataframe(
    job_df.iloc[top_jobs][['Job Title', 'Company Name', 'Location']]
)
