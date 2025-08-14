import os
import json
import joblib
import logging
import pandas as pd
import seaborn as sns
from typing import Tuple
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from assets.configs import ClassificationModelsConfig as Config


def setup_logger() -> logging.Logger:
    """Set up the logger."""
    logger = logging.getLogger(__name__)
    logger.setLevel(Config.LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


def splitData(X: pd.DataFrame, y: pd.Series) -> Tuple:
    """Split dataset into train, validation, and test sets"""
    logger.info("Splitting data into train, validation, and test sets...")

    x_train, x_temp, y_train, y_temp = train_test_split(
        X, y, stratify=y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=Config.RANDOM_STATE
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def scaleData(splitted_data: tuple):
    """Scale features using StandardScaler."""
    logger.info("Scaling features...")

    x_train, x_val, x_test, y_train, y_val, y_test = splitted_data

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test, scaler


def evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    """Evaluate model on train, validation, and test sets."""
    logger.info(f"Evaluating {model.__class__.__name__}...")

    def get_metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True
            ),
        }

    results = {
        "train": get_metrics(y_train, model.predict(x_train)),
        "validation": get_metrics(y_val, model.predict(x_val)),
        "test": get_metrics(y_test, model.predict(x_test)),
    }
    return results


def plot_confusion_matrix(y_true, y_pred, title, save_path) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results(
    model_name: str,
    model,
    results: dict,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    scaler: StandardScaler,
) -> None:
    """Save model, metrics, and scaler to disk."""
    model_dir = os.path.join(Config.RESULTS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    plot_confusion_matrix(
        y_train,
        model.predict(x_train),
        f"{model_name} - Train",
        os.path.join(model_dir, "confusion_matrix_train.png"),
    )
    plot_confusion_matrix(
        y_val,
        model.predict(x_val),
        f"{model_name} - Validation",
        os.path.join(model_dir, "confusion_matrix_val.png"),
    )
    plot_confusion_matrix(
        y_test,
        model.predict(x_test),
        f"{model_name} - Test",
        os.path.join(model_dir, "confusion_matrix_test.png"),
    )

    joblib.dump(model, os.path.join(model_dir, f"{model_name}.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

    logger.info(f"Saved results for {model_name} in {model_dir}")


def main():
    logger.info("Starting model training pipeline...")

    df = pd.read_excel("datasets/segmented_customers_dataset.xlsx").drop(
        columns=["Unnamed: 0"]
    )
    X = df.drop(columns=["CustomerSegmentation"])
    y = df["CustomerSegmentation"]

    splitted_data = splitData(X, y)
    x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test, scaler = (
        scaleData(splitted_data)
    )

    models = {
        "RandomForest": RandomForestClassifier(random_state=Config.RANDOM_STATE),
        "SVC": SVC(probability=True, random_state=Config.RANDOM_STATE),
    }

    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model.fit(x_train_scaled, y_train)

        results = evaluate_model(
            model, x_train_scaled, y_train, x_val_scaled, y_val, x_test_scaled, y_test
        )
        save_results(
            model_name,
            model,
            results,
            x_train_scaled,
            y_train,
            x_val_scaled,
            y_val,
            x_test_scaled,
            y_test,
            scaler,
        )

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
