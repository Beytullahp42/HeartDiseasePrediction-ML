from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import os
import sys

try:
    import train_model
except ImportError:
    print("Warning: 'train_model.py' not found.")
    train_model = None

app = Flask(__name__)
CORS(app)

MODEL_FILE = 'heart_disease_model.pkl'


def load_or_train_model():
    if not os.path.exists(MODEL_FILE):
        print(f"\nWarning: '{MODEL_FILE}' not found.")

        if train_model is None:
            print("Error: Model is missing AND 'train_model.py' is missing.")
            print("Please ensure files are in the same folder.")
            sys.exit(1)

        print("Initiating training sequence...")

        try:
            if hasattr(train_model, 'train_and_verify_model'):
                train_model.train_and_verify_model()
            else:
                print("Error: 'train_model.py' exists but has no recognized training function.")
                sys.exit(1)

        except Exception as e:
            print(f"Training failed: {e}")
            sys.exit(1)

    if os.path.exists(MODEL_FILE):
        print(f"Loading model: {MODEL_FILE}")
        return joblib.load(MODEL_FILE)
    else:
        print("Error: Training script ran, but model file is STILL missing.")
        sys.exit(1)


model = load_or_train_model()


@app.route('/')
def home():
    return send_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope'])
        ]

        final_features = [np.array(features)]
        prediction = model.predict(final_features)

        result = "Heart Disease Detected (High Risk)" if prediction[0] == 1 else "Healthy (Low Risk)"
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    if model is None:
        model = load_or_train_model()

    app.run(debug=True, port=5000)
