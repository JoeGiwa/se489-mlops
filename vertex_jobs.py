import subprocess

models = ["cnn", "mlp", "xgboost"]
region = "us-central1"
project_id = "utility-cumulus-462615-h4"
container_uri = f"us-central1-docker.pkg.dev/{project_id}/mlops-repo/mlops-train:latest"

for model in models:
    job_id = f"train-{model}-job"
    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        "--region",
        region,
        "--display-name",
        job_id,
        "--worker-pool-spec",
        f"machine-type=n1-standard-4,replica-count=1,container-image-uri={container_uri}",
        "--args",
        f"model={model}",
    ]
    print(f"\n🚀 Submitting Vertex AI training job for model: {model}")
    subprocess.run(cmd, check=True)
