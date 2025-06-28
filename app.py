from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")

# Homepage route (serves HTML form)
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not available"}), 500

    try:
        data = request.get_json()
        feature1 = data.get('feature1')
        feature2 = data.get('feature2')

        # Validate inputs
        if feature1 is None or feature2 is None:
            return jsonify({"error": "Missing 'feature1' or 'feature2'"}), 400

        # Predict
        input_data = np.array([[feature1, feature2]])
        prediction = model.predict(input_data)[0]

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run locally (Render uses gunicorn instead)
if __name__ == '__main__':
    app.run(debug=True)

