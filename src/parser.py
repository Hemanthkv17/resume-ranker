"""
parser.py
---------
Handles resume ingestion (PDF / DOCX / TXT) and rule-based feature
extraction: skills, years of experience, and education level.

Design choice: extraction here is deliberately keyword/regex based rather
than a trained NER model. See README "Tradeoffs" section for why.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    import docx  # python-docx
except ImportError:  # pragma: no cover
    docx = None


SKILLS_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "skills_db.txt")

EDUCATION_LEVELS = {
    "phd": 4,
    "doctorate": 4,
    "master": 3,
    "m.tech": 3,
    "m.sc": 3,
    "mba": 3,
    "msc": 3,
    "bachelor": 2,
    "b.tech": 2,
    "b.e": 2,
    "b.sc": 2,
    "bsc": 2,
    "be ": 2,
    "diploma": 1,
    "high school": 0,
}

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*years?\s+of\s+experience",
    r"(\d+)\+?\s*years?\s+experience",
    r"experience\s*[:\-]?\s*(\d+)\+?\s*years?",
]


@dataclass
class ResumeFeatures:
    source: str
    raw_text: str
    skills: List[str] = field(default_factory=list)
    experience_years: float = 0.0
    education_level: int = 0
    education_label: str = "unspecified"


def _load_skills_db() -> List[str]:
    with open(SKILLS_DB_PATH, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


_SKILLS_DB = _load_skills_db()


def extract_text(path: str) -> str:
    """Extract raw text from a .pdf, .docx, or .txt resume file."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("PyPDF2 is required to read PDF files. pip install PyPDF2")
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".docx":
        if docx is None:
            raise RuntimeError("python-docx is required to read DOCX files. pip install python-docx")
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs)

    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    raise ValueError(f"Unsupported resume file type: {ext}")


def extract_skills(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for skill in _SKILLS_DB:
        # word-boundary-ish match so "c" doesn't match inside "science"
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


def extract_experience_years(text: str) -> float:
    text_lower = text.lower()
    candidates = []
    for pattern in EXPERIENCE_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            try:
                candidates.append(float(match.group(1)))
            except (ValueError, IndexError):
                continue
    if candidates:
        return max(candidates)
    return 0.0


def extract_education_level(text: str):
    text_lower = text.lower()
    best_level = 0
    best_label = "unspecified"
    for label, level in EDUCATION_LEVELS.items():
        pattern = r"(?<![a-zA-Z])" + re.escape(label.strip()) + r"(?![a-zA-Z])"
        if re.search(pattern, text_lower) and level > best_level:
            best_level = level
            best_label = label
    return best_level, best_label


def parse_resume(path: str) -> ResumeFeatures:
    text = extract_text(path)
    skills = extract_skills(text)
    experience = extract_experience_years(text)
    edu_level, edu_label = extract_education_level(text)
    return ResumeFeatures(
        source=os.path.basename(path),
        raw_text=text,
        skills=skills,
        experience_years=experience,
        education_level=edu_level,
        education_label=edu_label,
    )
