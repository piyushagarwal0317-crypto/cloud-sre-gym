FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Setup for Hugging Face Space or standard execution
ENV PYTHONPATH=/app
CMD ["python", "-m", "tests.test_env"]