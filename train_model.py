import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / 'data' / 'heart_statlog_cleveland_hungary_final.csv'
MODEL_NAME = BASE_DIR / 'heart_disease_model.pkl'


def train_and_verify_model():
    print(f"Loading Data from {FILE_PATH}...")

    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {FILE_PATH}.")
        return

    rename_map = {
        'chest pain type': 'cp',
        'resting bp s': 'trestbps',
        'cholesterol': 'chol',
        'fasting blood sugar': 'fbs',
        'resting ecg': 'restecg',
        'max heart rate': 'thalach',
        'exercise angina': 'exang',
        'ST slope': 'slope'
    }
    df = df.rename(columns=rename_map)

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf_model.predict(X_train))
    test_acc = accuracy_score(y_test, rf_model.predict(X_test))

    print("\n" + "=" * 40)
    print(f"🔍 MODEL VERIFICATION")
    print("=" * 40)
    print(f"   Training Accuracy: {train_acc * 100:.2f}%")
    print(f"   Test Accuracy:     {test_acc * 100:.2f}% (Should be 94.54%)")
    print("=" * 40)

    joblib.dump(rf_model, MODEL_NAME)
    print(f"\nVerified Model saved as '{MODEL_NAME}'")

if __name__ == "__main__":
    train_and_verify_model()
