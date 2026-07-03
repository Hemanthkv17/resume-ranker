"""
features.py
-----------
Builds the fixed-length feature vector fed into the ranking model,
combining NLP similarity with the rule-based resume features.
"""

from typing import List
import numpy as np

from src.parser import ResumeFeatures
from src.similarity import skill_overlap_ratio

FEATURE_NAMES = [
    "tfidf_similarity",
    "skill_overlap_ratio",
    "experience_match_ratio",
    "education_match_ratio",
]


def experience_match_ratio(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 1.0
    ratio = candidate_years / required_years
    return float(min(ratio, 1.5) / 1.5)  # cap so 1.5x+ the requirement maxes out at 1.0


def education_match_ratio(candidate_level: int, required_level: int) -> float:
    if required_level <= 0:
        return 1.0
    return float(min(candidate_level / required_level, 1.0))


def build_feature_vector(
    resume: ResumeFeatures,
    jd_skills: List[str],
    jd_experience_years: float,
    jd_education_level: int,
    tfidf_score: float,
) -> np.ndarray:
    return np.array([
        tfidf_score,
        skill_overlap_ratio(resume.skills, jd_skills),
        experience_match_ratio(resume.experience_years, jd_experience_years),
        education_match_ratio(resume.education_level, jd_education_level),
    ])
