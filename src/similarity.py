"""
similarity.py
-------------
NLP-based similarity between a resume and a job description.

Design choice: TF-IDF + cosine similarity rather than a transformer
embedding model. See README "Tradeoffs" section for the reasoning
(no external model download required, fast, fully deterministic,
still reasonably strong for keyword-dense documents like resumes/JDs).
"""

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity(job_description: str, resumes: List[str]) -> List[float]:
    """
    Returns a cosine similarity score (0-1) between the job description
    and each resume, using a shared TF-IDF vector space so scores are
    comparable across candidates.
    """
    corpus = [job_description] + resumes
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    sims = cosine_similarity(jd_vector, resume_vectors)[0]
    return sims.tolist()


def skill_overlap_ratio(resume_skills: List[str], jd_skills: List[str]) -> float:
    """Fraction of JD-required skills that are present in the resume."""
    if not jd_skills:
        return 0.0
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)
    matched = resume_set & jd_set
    return len(matched) / len(jd_set)
