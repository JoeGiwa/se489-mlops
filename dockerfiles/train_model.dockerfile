# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_randproject/ mlops_randproject/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_randproject/models/model_training.py", "model=mlp", "train.epochs=2"]

# ENTRYPOINT ["python", "-u", "mlops_randproject/models/model_training.py"]
