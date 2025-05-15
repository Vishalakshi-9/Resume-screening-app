import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def rank_resumes(resume_texts, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_description] + resume_texts)
    jd_vector = vectors[0]
    resume_vectors = vectors[1:]
    similarities = cosine_similarity(jd_vector, resume_vectors).flatten()
    ranked_indices = similarities.argsort()[::-1]
    return [(idx, similarities[idx]) for idx in ranked_indices]

def detect_job_role(resume_text, job_roles_dict):
    roles = list(job_roles_dict.keys())
    descriptions = list(job_roles_dict.values())
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text] + descriptions)
    resume_vec = vectors[0:1]
    role_vecs = vectors[1:]
    similarities = cosine_similarity(resume_vec, role_vecs).flatten()
    best_index = similarities.argmax()
    return roles[best_index], similarities[best_index]
