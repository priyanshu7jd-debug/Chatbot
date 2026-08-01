"""
Evaluate the trained category classifier: accuracy, classification report,
and a confusion matrix plot.

Usage:
    python -m src.evaluate
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.train import load_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    import joblib

    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "nb_model.pkl"))

    df = load_data()
    X = vectorizer.transform(df["cleaned_questions"])
    y = df["category"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted category")
    plt.ylabel("Actual category")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(label="Number of questions")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "confusion_matrix.png")
    plt.savefig(out_path)
    print(f"Saved confusion matrix plot to {out_path}")


if __name__ == "__main__":
    main()
