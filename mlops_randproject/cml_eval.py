# mlops_randproject/run_cml_eval.py

import os
import argparse
import numpy as np
from mlops_randproject.evaluate import evaluate_model

if os.path.exists("metrics_report.md") and os.getenv("CML_FIRST_MODEL") == "true":
    os.remove("metrics_report.md")


def main(model_name):
    predictions_path = f"artifacts/predictions_{model_name}.txt"
    true_labels_path = "artifacts/test_labels.npy"
    cm_image_path = "artifacts/confusion_matrix.png"

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not os.path.exists(true_labels_path):
        raise FileNotFoundError(f"True labels file not found: {true_labels_path}")

    preds = np.loadtxt(predictions_path, dtype=int)
    true_labels = np.load(true_labels_path)

    metrics = evaluate_model(preds, true_labels)

    report_path = "metrics_report.md"
    with open("metrics_report.md", "a") as f:
        f.write(f"\n## {model_name.upper()} Evaluation Metrics\n")
        f.write(f"- Accuracy: **{metrics['accuracy']:.4f}**\n")
        f.write(f"- F1 Score: **{metrics['f1_score']:.4f}**\n")
        f.write(f"![Confusion Matrix](artifacts/confusion_matrix_{model_name}.png)\n")

        if os.path.exists(cm_image_path):
            f.write("### 🔍 Confusion Matrix\n\n")
            f.write(f"![Confusion Matrix]({cm_image_path})\n")

    print(f"✅ CML report generated: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions and generate CML report."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (cnn/mlp/xgboost)"
    )
    args = parser.parse_args()
    main(args.model)
