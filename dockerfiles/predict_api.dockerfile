# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project code and artifacts
COPY mlops_randproject/ mlops_randproject/
COPY artifacts/ artifacts/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app from correct location
CMD ["uvicorn", "mlops_randproject.api.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
