"""
rank.py
-------
End-to-end pipeline: parse job description + resumes, extract features,
score with the ML model, and output a ranked list of candidates.
"""

import os
import glob
import argparse
import json
import time
from src.parser import parse_resume, extract_skills, extract_experience_years, extract_education_level
from src.similarity import tfidf_similarity
from src.features import build_feature_vector
from src.model import load_model, predict_score


def load_job_description(jd_path: str):
    with open(jd_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    skills = extract_skills(text)
    experience_years = extract_experience_years(text)
    education_level, education_label = extract_education_level(text)
    return {
        "text": text,
        "skills": skills,
        "experience_years": experience_years,
        "education_level": education_level,
        "education_label": education_label,
    }


def rank_candidates(jd_path: str, resumes_dir: str):
    jd = load_job_description(jd_path)

    resume_paths = sorted(
        p for p in glob.glob(os.path.join(resumes_dir, "*"))
        if os.path.splitext(p)[1].lower() in (".pdf", ".docx", ".txt")
    )
    if not resume_paths:
        raise FileNotFoundError(f"No .pdf/.docx/.txt resumes found in {resumes_dir}")

    parsed_resumes = [parse_resume(p) for p in resume_paths]
    tfidf_scores = tfidf_similarity(jd["text"], [r.raw_text for r in parsed_resumes])

    model = load_model()

    results = []
    for resume, tfidf_score in zip(parsed_resumes, tfidf_scores):
        feature_vector = build_feature_vector(
            resume=resume,
            jd_skills=jd["skills"],
            jd_experience_years=jd["experience_years"],
            jd_education_level=jd["education_level"],
            tfidf_score=tfidf_score,
        )
        score = predict_score(model, feature_vector)
        results.append({
            "candidate": resume.source,
            "fit_score": round(score, 2),
            "tfidf_similarity": round(tfidf_score, 3),
            "matched_skills": sorted(set(resume.skills) & set(jd["skills"])),
            "missing_skills": sorted(set(jd["skills"]) - set(resume.skills)),
            "experience_years": resume.experience_years,
            "education": resume.education_label,
        })

    results.sort(key=lambda r: r["fit_score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    return {
        "job_description": {
            "required_skills": jd["skills"],
            "required_experience_years": jd["experience_years"],
            "required_education": jd["education_label"],
        },
        "ranking": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Resume Screening and Ranking System using NLP and Machine Learning"
    )

    parser.add_argument(
        "--jd",
        required=True,
        help="Path to the Job Description (.txt)"
    )

    parser.add_argument(
        "--resumes",
        required=True,
        help="Folder containing resumes (.pdf/.docx/.txt)"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file"
    )

    args = parser.parse_args()

    start_time = time.time()

    output = rank_candidates(args.jd, args.resumes)

    print("\n" + "=" * 70)
    print("        AI Resume Screening and Ranking System")
    print("=" * 70)

    print("\nJob Description Loaded Successfully")
    print(f"Total Resumes Processed : {len(output['ranking'])}")
    print("=" * 70)

    for r in output["ranking"]:

        print(f"\nRank #{r['rank']}")
        print("-" * 50)

        print(f"Candidate Name      : {r['candidate']}")
        print(f"Final Score         : {r['fit_score']:.2f}")
        print(f"Experience          : {r['experience_years']} Years")
        print(f"Education           : {r['education']}")
        print(f"TF-IDF Similarity   : {r['tfidf_similarity']}")

        print("\nMatched Skills")
        print("-" * 20)

        if r["matched_skills"]:
            for skill in r["matched_skills"]:
                print(f"✓ {skill}")
        else:
            print("No matched skills found.")

        print("\nMissing Skills")
        print("-" * 20)

        if r["missing_skills"]:
            for skill in r["missing_skills"]:
                print(f"✗ {skill}")
        else:
            print("No missing skills.")

        print("=" * 70)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print("\nResults saved successfully.")
        print(f"Output File : {args.output}")

    end_time = time.time()

    print(f"\nExecution Time : {end_time - start_time:.2f} seconds")
    print("=" * 70)