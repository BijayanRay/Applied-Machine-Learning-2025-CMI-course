import joblib
import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import expit
from typing import Tuple

# Load the trained model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("support_vector_machine_best_model.pkl")

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    if not isinstance(text, str):  # Ensures input is a string
        raise ValueError("Input text must be a string.")
    if not text.strip():  # Ensure non-empty text
        raise ValueError("Input text cannot be empty.")
    text_vectorized = vectorizer.transform([text]) # vectorizer
    decision_score = model.decision_function(text_vectorized) # get raw scores
    propensity = expit(decision_score[0]) # decision scores to probability
    prediction = propensity >= threshold # threshold propensity
    return bool(prediction), float(propensity)
