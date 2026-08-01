"""
Basic tests for the chatbot pipeline.

Run with:
    python -m unittest discover tests
(after running `python -m src.train` at least once to generate models/)
"""

import os
import unittest

from src.chatbot import BankingChatbot
from src.preprocess import clean_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class TestCleanText(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(clean_text("What is the CAPITAL of France?"), "capital france")

    def test_removes_stopwords(self):
        cleaned = clean_text("What is my current account balance?")
        self.assertNotIn("is", cleaned.split())
        self.assertNotIn("my", cleaned.split())
        self.assertIn("balance", cleaned.split())


@unittest.skipUnless(
    os.path.exists(os.path.join(MODELS_DIR, "nb_model.pkl")),
    "Run `python -m src.train` first to generate trained model artifacts.",
)
class TestBankingChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = BankingChatbot()

    def test_exact_known_question_returns_matching_response(self):
        reply = self.bot.get_response("I forgot my online banking password")
        self.assertEqual(reply.similarity, 1.0)
        self.assertIn("password", reply.response.lower())

    def test_unrelated_question_triggers_fallback(self):
        reply = self.bot.get_response("what's the weather like today")
        self.assertLess(reply.similarity, self.bot.threshold)
        self.assertIn("human agent", reply.response.lower())


if __name__ == "__main__":
    unittest.main()
