"""
Inference-time chatbot logic.

Category prediction alone isn't precise enough to pick a reply (one category
can map to several different responses in the training data), so the bot
combines two signals:

1. A Naive Bayes model predicts the most likely category (for context/logging).
2. A cosine-similarity search over TF-IDF vectors finds the single most similar
   question ever seen, and reuses that question's real response.

If nothing in the training data is similar enough, the bot admits it isn't
confident instead of guessing.
"""

import os
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import clean_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
FALLBACK_RESPONSE = (
    "I'm not confident I understood that correctly. "
    "Let me connect you with a human agent who can help."
)


@dataclass
class BotReply:
    response: str
    predicted_category: str
    category_confidence: float
    matched_question: str
    similarity: float


class BankingChatbot:
    """Loads trained artifacts and answers customer questions."""

    def __init__(self, models_dir: str = MODELS_DIR, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
        model_path = os.path.join(models_dir, "nb_model.pkl")
        data_path = os.path.join(models_dir, "reference_data.csv")

        if not (os.path.exists(vectorizer_path) and os.path.exists(model_path) and os.path.exists(data_path)):
            raise FileNotFoundError(
                "Trained model artifacts not found. Run `python -m src.train` first "
                f"to generate them in {models_dir}/."
            )

        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        self.reference_data: pd.DataFrame = pd.read_csv(data_path)
        self.reference_vectors = self.vectorizer.transform(self.reference_data["cleaned_questions"])
        self.threshold = threshold

    def get_response(self, user_question: str) -> BotReply:
        cleaned = clean_text(user_question)
        vectorized = self.vectorizer.transform([cleaned])

        predicted_category = self.model.predict(vectorized)[0]
        category_confidence = float(self.model.predict_proba(vectorized)[0].max())

        similarities = cosine_similarity(vectorized, self.reference_vectors)[0]
        best_idx = similarities.argmax()
        best_similarity = float(similarities[best_idx])

        if best_similarity < self.threshold:
            response = FALLBACK_RESPONSE
        else:
            response = self.reference_data.iloc[best_idx]["response"]

        return BotReply(
            response=response,
            predicted_category=predicted_category,
            category_confidence=round(category_confidence, 2),
            matched_question=self.reference_data.iloc[best_idx]["questions"],
            similarity=round(best_similarity, 2),
        )
