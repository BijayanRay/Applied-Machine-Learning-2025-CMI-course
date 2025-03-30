import joblib
from flask import Flask, request, jsonify, render_template_string
from score import score

app = Flask(__name__)

# Load the trained model
model = joblib.load("support_vector_machine_best_model.pkl")

# html part to display in the browser
html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Message Classifier</title>
    <style>
        body {margin: 60px; text-align: center; }
        h1 { color: #333; }
        form { margin-top: 20px; }
        input[type="text"] { width: 60%; padding: 10px; margin: 10px 0; font-size: 16px; }
        button { padding: 10px 15px; font-size: 16px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .result { margin-top: 20px; font-size: 18px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Message Classifier</h1>
    <form method="post">
        <input type="text" name="text" placeholder="Enter text to classify" required>
        <button type="submit">Classify</button>
    </form>
    {% if result %}
        <p class="result">{{ result }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def classify(): # classify
    result = None
    if request.method == "POST":
        text = request.form.get("text", "").strip() # input text
        if text:
            prediction, propensity = score(text, model, threshold=0.5) # get the prediction and propensity
            result = f"Prediction: {'Spam' if prediction else 'Not Spam'} (Propensity: {propensity:.2f})" # output the prediction and propensity
    return render_template_string(html, result=result)

@app.route("/score", methods=["POST"])
def score_endpoint(): # get the scores
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    prediction, propensity = score(text, model, threshold=0.5) # get the prediction and propensity
    return jsonify({"prediction": prediction, "propensity": propensity}) # return the json

if __name__ == "__main__":
    app.run()
