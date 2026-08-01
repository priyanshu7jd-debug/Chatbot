"""
Train the banking support chatbot.

Loads data/banking_support.csv, cleans the questions, fits a TF-IDF
vectorizer + Multinomial Naive Bayes classifier, evaluates on a held-out
split, and saves all the artifacts needed for inference to models/.

Usage:
    python -m src.train
"""

import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.preprocess import clean_text

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "banking_support.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cleaned_questions"] = df["questions"].apply(clean_text)
    return df


def train(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df["cleaned_questions"])
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy * 100:.1f}% ({len(y_test)} test examples)")
    print(classification_report(y_test, y_pred, zero_division=0))

    return vectorizer, model, X


def save_artifacts(vectorizer, model, df, models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
    joblib.dump(model, os.path.join(models_dir, "nb_model.pkl"))
    df.to_csv(os.path.join(models_dir, "reference_data.csv"), index=False)
    print(f"Saved vectorizer.pkl, nb_model.pkl, and reference_data.csv to {models_dir}/")


def main():
    df = load_data()
    print(f"Loaded {len(df)} tickets across {df['category'].nunique()} categories")
    vectorizer, model, _ = train(df)
    save_artifacts(vectorizer, model, df)


if __name__ == "__main__":
    main()
