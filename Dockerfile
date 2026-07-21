FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app.py .
COPY index.html .
COPY train_model.py .
COPY heart_disease_model.pkl .
RUN python -c "import warnings, joblib; from sklearn.exceptions import InconsistentVersionWarning; warnings.simplefilter('error', InconsistentVersionWarning); joblib.load('heart_disease_model.pkl')"
COPY data ./data

EXPOSE 8437

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8437} app:app"]
