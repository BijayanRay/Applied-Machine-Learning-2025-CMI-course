import joblib
import multiprocessing
import os
import requests
import time
import pytest
from score import score
import app
import subprocess

def test_score():
    """Unit tests for the score function."""
    model = joblib.load("support_vector_machine_best_model.pkl")

    # Smoke test (ensures function runs without crashing)
    text = "This is a test message."
    threshold = 0.5
    result = score(text, model, threshold)
    assert isinstance(result, tuple), "Output should be a tuple"
    assert len(result) == 2, "Output tuple should contain two elements"

    # Format test
    prediction, propensity = result
    assert isinstance(prediction, bool), "Prediction should be a boolean"
    assert isinstance(propensity, float), "Propensity should be a float"

    # Range test
    assert 0.0 <= propensity <= 1.0, "Propensity should be between 0 and 1"

    # Threshold 0 should always return True (spam)
    assert score(text, model, 0.0)[0] == True, "With threshold 0, prediction should always be 1"

    # Threshold 1 should always return False (not spam)
    assert score(text, model, 1.0)[0] == False, "With threshold 1, prediction should always be 0"

    # Obvious spam test
    spam_text = "Great News! Call FREEFONE 08006344447 to claim your guaranteed £1000 CASH or £2000 gift. Speak to a live operator NOW!"
    assert score(spam_text, model, 0.5)[0] == True, "Obvious spam should return 1"

    # Obvious non-spam test
    non_spam_text = "Hey, let's meet for tennis tomorrow."
    assert score(non_spam_text, model, 0.5)[0] == False, "Obvious non-spam should return 0"

    # Edge cases
    try:
        score(12345, model, 0.5)
    except ValueError as e:
        assert str(e) == "Input text must be a string.", "Should raise ValueError for non-string input"

    try:
        score("", model, 0.5)
    except ValueError as e:
        assert str(e) == "Input text cannot be empty.", "Should raise ValueError for empty string input"

    print("All unit tests passed.")

def run_flask():
    """Runs the Flask app."""
    os.system("python app.py")

def test_flask():
    """Integration test for Flask API."""
    flask_process = multiprocessing.Process(target=run_flask)
    flask_process.start()
    time.sleep(3)  # Allow some time for the server to start

    try:
        url = "http://127.0.0.1:5000/score"

        # Valid request
        payload = {"text": "You have won a free iPhone!"}
        response = requests.post(url, json=payload)
        data = response.json()

        assert isinstance(data, dict), "Response should be a dictionary"
        assert "prediction" in data and "propensity" in data, "Response should contain 'prediction' and 'propensity'"
        assert isinstance(data["prediction"], bool), "Prediction should be a boolean"
        assert isinstance(data["propensity"], float), "Propensity should be a float"
        assert 0.0 <= data["propensity"] <= 1.0, "Propensity should be between 0 and 1"

        # Test missing input
        response = requests.post(url, json={})
        assert response.status_code == 400, "Should return 400 for missing text"

        # Test empty text
        response = requests.post(url, json={"text": ""})
        assert response.status_code == 400, "Should return 400 for empty text"

        print("Integration test passed.")

    finally:
        flask_process.terminate()
        flask_process.join()
CONTAINER_NAME = "flask-container"

def test_docker():
    # Build the Docker image
    build_cmd = "docker build -t flask-app ."
    assert subprocess.run(build_cmd, shell=True).returncode == 0

    # Run the Docker container
    run_cmd = f"docker run -d -p 5000:5000 --name {CONTAINER_NAME} flask-app"
    assert subprocess.run(run_cmd, shell=True).returncode == 0

    # Wait for the container to start
    time.sleep(5)

    # Send a request to the /score endpoint
    url = "http://127.0.0.1:5000/score"
    sample_text = {"text": "This is a test"}

    try:
        response = requests.post(url, json=sample_text)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict), "Response should be a dictionary"
        assert "prediction" in data and "propensity" in data, "Response should contain 'prediction' and 'propensity'"
        assert isinstance(data["prediction"], bool), "Prediction should be a boolean"
        assert isinstance(data["propensity"], float), "Propensity should be a float"
        assert 0.0 <= data["propensity"] <= 1.0, "Propensity should be between 0 and 1"

    finally:
        # Stop and remove the container
        subprocess.run(f"docker stop {CONTAINER_NAME}", shell=True)
        subprocess.run(f"docker rm {CONTAINER_NAME}", shell=True)

if __name__ == "__main__":
    test_score()
    test_flask()
    test_docker()
