FROM python:3.11.18-slim
WORKDIR /work
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/cache/hf
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY app/ app/
COPY scripts/ scripts/
COPY tests/ tests/
COPY pytest.ini .
