"""
Trains the GradientBoostingRegressor ranking model on synthetic labeled
data and saves it to models/ranker_model.pkl.

Usage:
    python train_model.py
"""
from src.model import train_model

if __name__ == "__main__":
    train_model(save=True)
