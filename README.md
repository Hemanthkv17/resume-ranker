# Resume Screening and Ranking System

## Overview

This project is a Resume Screening and Ranking System developed using Python, NLP, and Machine Learning. It helps automate the initial screening process by comparing resumes with a given job description and ranking candidates based on how well they match the required skills and qualifications.

The system extracts important information such as skills, work experience, and education from resumes, calculates their similarity with the job description, and generates a final ranking score using a machine learning model.

---

## Features

- Extracts text from PDF, DOCX, and TXT resumes
- Identifies candidate skills, experience, and education
- Compares resumes with the job description using TF-IDF and Cosine Similarity
- Uses a Gradient Boosting model to generate a final candidate score
- Displays ranked candidates along with matched and missing skills
- Saves the complete ranking results in a JSON file

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- PyPDF2
- python-docx

---

## Project Structure

```
resume-ranker
│
├── main.py
├── requirements.txt
├── README.md
├── sample_data
│   ├── job_description.txt
│   └── resumes
│
├── models
│   └── ranker_model.pkl
│
├── src
│   ├── parser.py
│   ├── features.py
│   ├── similarity.py
│   ├── model.py
│   └── rank.py
│
└── tests
```

---

## How the Project Works

1. Read the job description.
2. Read all resumes from the selected folder.
3. Extract candidate information such as:
   - Skills
   - Experience
   - Education
4. Calculate similarity between each resume and the job description using TF-IDF and Cosine Similarity.
5. Generate feature values for each candidate.
6. Pass the features to the trained Machine Learning model.
7. Calculate a final score.
8. Rank all candidates from highest to lowest score.
9. Save the results in a JSON file.

---

## Installation

Clone the repository.

```bash
git clone <repository-url>
cd resume-ranker
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

Windows

```bash
venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the project using:

```bash
python main.py --jd sample_data/job_description.txt --resumes sample_data/resumes --output results.json
```

After execution, the ranked candidates will be displayed in the terminal and stored in:

```
results.json
```

---

## Sample Output

```
=========== Candidate Ranking ===========

1. Priya Sharma
Score : 69.49

2. Rohit Verma
Score : 57.12

3. Kavya Iyer
Score : 27.58

4. Arjun Mehta
Score : 24.89
```

---

## Model Used

The project uses a **Gradient Boosting Regressor** to predict the final candidate score.

The model considers:

- Skill Match
- Resume Similarity
- Years of Experience
- Education Level

These features together help generate a balanced ranking instead of relying on a single parameter.

---

## Limitations

- The ranking model is trained using synthetic data because real hiring datasets are not publicly available.
- Skill extraction is based on predefined keywords.
- TF-IDF cannot fully understand the meaning of sentences like transformer-based language models.

---

## Future Improvements

Some possible enhancements include:

- Resume upload through a web interface
- BERT/Sentence Transformer embeddings
- Better skill extraction using Named Entity Recognition
- Dashboard for recruiters
- Support for multiple job descriptions
- Resume recommendation analytics

---

## Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can be used to automate the resume screening process. It provides a simple, explainable, and scalable solution for ranking candidates based on their relevance to a job description.