import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parser import extract_skills, extract_experience_years, extract_education_level
from src.similarity import tfidf_similarity, skill_overlap_ratio
from src.features import experience_match_ratio, education_match_ratio
from src.model import generate_synthetic_training_data, train_model, predict_score


class TestParser(unittest.TestCase):
    def test_extract_skills_finds_known_skills(self):
        text = "Experienced in Python, Docker, and AWS deployments."
        skills = extract_skills(text)
        self.assertIn("python", skills)
        self.assertIn("docker", skills)
        self.assertIn("aws", skills)

    def test_extract_skills_no_partial_match(self):
        # "c" should not match inside "science" or "docker"
        text = "Computer science and docker experience."
        skills = extract_skills(text)
        self.assertNotIn("c", skills)

    def test_extract_experience_years(self):
        text = "I have 5 years of experience in software engineering."
        self.assertEqual(extract_experience_years(text), 5.0)

    def test_extract_experience_years_missing(self):
        text = "No experience mentioned here."
        self.assertEqual(extract_experience_years(text), 0.0)

    def test_extract_education_level(self):
        text = "Bachelor of Technology in Computer Science."
        level, label = extract_education_level(text)
        self.assertEqual(level, 2)
        self.assertIn(label, ("bachelor", "b.tech"))


class TestSimilarity(unittest.TestCase):
    def test_tfidf_similarity_identical_text_is_high(self):
        jd = "Python backend engineer with REST API experience"
        sims = tfidf_similarity(jd, [jd, "Completely unrelated gardening and cooking text"])
        self.assertGreater(sims[0], sims[1])

    def test_skill_overlap_ratio(self):
        ratio = skill_overlap_ratio(["python", "sql"], ["python", "sql", "docker", "aws"])
        self.assertAlmostEqual(ratio, 0.5)

    def test_skill_overlap_ratio_no_jd_skills(self):
        self.assertEqual(skill_overlap_ratio(["python"], []), 0.0)


class TestFeatures(unittest.TestCase):
    def test_experience_match_ratio_caps_at_one(self):
        self.assertEqual(experience_match_ratio(10, 2), 1.0)

    def test_experience_match_ratio_partial(self):
        self.assertAlmostEqual(experience_match_ratio(1, 2), (1 / 2) / 1.5)

    def test_education_match_ratio_no_requirement(self):
        self.assertEqual(education_match_ratio(0, 0), 1.0)


class TestModel(unittest.TestCase):
    def test_synthetic_data_shapes(self):
        X, y = generate_synthetic_training_data(n_samples=50)
        self.assertEqual(X.shape, (50, 4))
        self.assertEqual(y.shape, (50,))
        self.assertTrue((y >= 0).all() and (y <= 100).all())

    def test_train_and_predict(self):
        model = train_model(save=False)
        import numpy as np
        strong_candidate = np.array([0.9, 0.9, 0.9, 0.9])
        weak_candidate = np.array([0.1, 0.1, 0.1, 0.1])
        strong_score = predict_score(model, strong_candidate)
        weak_score = predict_score(model, weak_candidate)
        self.assertGreater(strong_score, weak_score)


if __name__ == "__main__":
    unittest.main()
