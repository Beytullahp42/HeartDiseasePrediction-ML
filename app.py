from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'heart_disease_model.pkl')
INDEX_FILE = os.path.join(BASE_DIR, 'index.html')

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(os.path.join(BASE_DIR, '.env'))

try:
    import train_model
except ImportError:
    print("Warning: 'train_model.py' not found.")
    train_model = None

app = Flask(__name__)

cors_origins = [
    origin.strip()
    for origin in os.environ.get('CORS_ORIGINS', '').split(',')
    if origin.strip()
]

if cors_origins:
    CORS(app, resources={r"/predict": {"origins": cors_origins}})


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
    return send_file(INDEX_FILE)


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

    port = int(os.environ.get('PORT', 8437))
    app.run(debug=True, host='0.0.0.0', port=port)
