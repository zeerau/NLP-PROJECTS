# AI Resume Screener (Streamlit App)
# This code creates a resume screening app using TF-IDF + Logistic Regression with multi-resume ranking

import streamlit as st
import pandas as pd
import PyPDF2
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load or train model (for demo purpose, training is done inline)
@st.cache_data
def load_data():
    data = pd.read_csv("data/resume_dataset.csv")
    return data

def train_model(data):
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    pipe.fit(data["resume_text"], data["label"])
    return pipe

data = load_data()
model = train_model(data)

# Streamlit UI
st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🧠 AI Resume Screener")
st.write("Upload one or more resumes (PDF or TXT), and the AI will predict the most relevant job role for each.")

# File uploader
uploaded_files = st.file_uploader("Upload resume(s)", type=["pdf", "txt"], accept_multiple_files=True)

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        return str(file.read(), 'utf-8')

if uploaded_files:
    results = []
    for file in uploaded_files:
        resume_text = extract_text(file)
        prediction = model.predict([resume_text])[0]
        proba = model.predict_proba([resume_text]).max()
        results.append((file.name, prediction, proba))

    st.subheader("📊 Resume Screening Results")
    df_results = pd.DataFrame(results, columns=["Filename", "Predicted Role", "Confidence"])
    df_results.sort_values(by="Confidence", ascending=False, inplace=True)
    st.dataframe(df_results.style.format({"Confidence": "{:.2%}"}))
