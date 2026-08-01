"""
Command-line interface for the banking support chatbot.

Usage:
    python app.py
"""

from src.chatbot import BankingChatbot


def main():
    try:
        bot = BankingChatbot()
    except FileNotFoundError as e:
        print(e)
        return

    print("Banking Support Bot - type 'quit' to exit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Bot: Thanks for chatting - goodbye!")
            break
        if not user_input:
            continue
        reply = bot.get_response(user_input)
        print("Bot:", reply.response)
        print(f"     (category: {reply.predicted_category}, "
              f"confidence: {reply.category_confidence * 100:.0f}%, "
              f"match similarity: {reply.similarity * 100:.0f}%)")
        print()


if __name__ == "__main__":
    main()
