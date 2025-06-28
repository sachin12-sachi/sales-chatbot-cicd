from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['feature1'], data['feature2']]).reshape(1, -1)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(features)[0]
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
