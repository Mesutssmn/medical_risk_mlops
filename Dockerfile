# ── Build stage ──
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ──
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Set MLflow tracking URI
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Copy project
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY streamlit_app.py .
COPY .streamlit/ .streamlit/

# Expose both ports: 8000 (FastAPI) & 8501 (Streamlit)
EXPOSE 8000 8501
