# Heart Disease Prediction (ML)

This is a machine learning application built to predict the risk of cardiovascular disease (CVD) using patient physiological data. It was developed as a final project to reproduce and extend the findings of the 2022 IEEE paper _"Machine Learning Based Heart Disease Prediction: A Study for Home Personalized Care."_

The project benchmarks 7 supervised learning algorithms across multiple datasets. This implementation achieved a project-high **95.1% accuracy** using a Random Forest classifier on a larger Kaggle dataset, which is then deployed via a Flask REST API.

---

## 🛠️ Technologies Used

- **Python 3.12**
- **Scikit-learn** (Random Forest, SVM, KNN, Naive Bayes, etc.)
- **Flask**
- **XGBoost**
- **Pandas & NumPy**

---

## ⚙️ Web App Installation & Usage

### 1. Clone the Repository

```
git clone https://github.com/Beytullahp42/HeartDiseasePrediction-ML.git
cd HeartDiseasePrediction-ML
```

### 2. Set Up Virtual Environment

**Windows:**

```
python -m venv venv
venv\Scripts\activate

```

**Mac/Linux:**

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Configure Environment

Create a local environment file from the example:

```
cp .env.example .env
```

Update `.env` if you need different ports or allowed CORS origins.

### 5. Run the Application

```
python app.py
```

_Note: If `heart_disease_model.pkl` is not found, the app will automatically run `train_model.py` to train and verify the model before starting the server._

Access the web interface at: `http://127.0.0.1:8437/`

---

## 🐳 Docker Usage

Create or update `.env` before running Docker:

```
cp .env.example .env
```

Build the Docker image from the project folder:

```
docker build -t heart-disease-prediction .
```

Run the container:

```
docker run --rm --env-file .env -p 8437:8437 heart-disease-prediction
```

Open the web interface at: `http://127.0.0.1:8437/`

### Docker Compose

Build and run the app with Compose:

```
docker compose up --build
```

If port `8437` is already in use, choose another host port:

```
HOST_PORT=5050 docker compose up --build
```

Then open: `http://127.0.0.1:5050/`

---

## 📓 Research Notebook

For a deep dive into the experimental loop, data visualization, and benchmarking of all 7 algorithms, please refer to **`42x4.ipynb`**.

This notebook contains:

- The complete comparison of 6 dataset variations vs 7 models

- Confusion Matrices and Learning Curves

- The logic behind the "Imputation vs. Deletion" data engineering strategy

---

## ⚠️ Disclaimer

This application is for educational and research purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment.
