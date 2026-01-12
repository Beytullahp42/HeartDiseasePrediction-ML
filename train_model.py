import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
FILE_PATH = 'data/heart_statlog_cleveland_hungary_final.csv'
MODEL_NAME = 'heart_disease_model.pkl'


def train_and_verify_model():
    print(f"Loading Data from {FILE_PATH}...")

    # 1. Load Data
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {FILE_PATH}. Check folder structure.")
        return

    # 2. Rename columns (To standard format)
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

    # 3. Preprocessing (Numeric conversion)
    df = pd.get_dummies(df, drop_first=True)

    # 4. Split Features and Target
    X = df.drop('target', axis=1)
    y = df['target']

    # 5. Split Train/Test (Crucial to reproduce the 94.5% score)
    # We use random_state=42 exactly like in the notebook
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train the Model
    print("Training Random Forest Model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 7. CHECK ACCURACY (The step you asked for!)
    train_acc = accuracy_score(y_train, rf_model.predict(X_train))
    test_acc = accuracy_score(y_test, rf_model.predict(X_test))

    print("\n" + "=" * 40)
    print(f"🔍 MODEL VERIFICATION")
    print("=" * 40)
    print(f"   Training Accuracy: {train_acc * 100:.2f}%")
    print(f"   Test Accuracy:     {test_acc * 100:.2f}%  <-- (Should be ~94.54%)")
    print("=" * 40)

    # 8. Save to Disk
    joblib.dump(rf_model, MODEL_NAME)
    print(f"\n✅ SUCCESS! Verified Model saved as '{MODEL_NAME}'")
    print("You can now run 'python app.py'")


if __name__ == "__main__":
    train_and_verify_model()