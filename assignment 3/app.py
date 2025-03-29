import joblib
import json
from flask import Flask, request, jsonify
from score import score

app = Flask(__name__)

# Load the model once to avoid reloading for every request
model = joblib.load("support_vector_machine_best_model.pkl")

@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400 # jsonify the error and return
    prediction, propensity = score(text, model, threshold=0.5)
    return jsonify({"prediction": prediction, "propensity": propensity}) # jsonify the prediction and return

if __name__ == "__main__":
    app.run()
