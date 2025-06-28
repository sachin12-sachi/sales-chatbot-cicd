from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model once when the server starts
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"❌ Failed to load model: {e}")

@app.route('/')
def home():
    return "✅ Sales Chatbot API is live! Use /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        feature1 = data.get('feature1')
        feature2 = data.get('feature2')

        if feature1 is None or feature2 is None:
            return jsonify({"error": "Missing 'feature1' or 'feature2' in request"}), 400

        input_data = np.array([[feature1, feature2]])
        prediction = model.predict(input_data)[0]

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
