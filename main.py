"""
Entry point.

Usage:
    python main.py --jd sample_data/job_description.txt --resumes sample_data/resumes --output results.json
"""
from src.rank import main

if __name__ == "__main__":
    main()
