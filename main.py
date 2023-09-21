import json
import random

# Load intents from intents.json
with open("intents.json", "r") as file:
    intents = json.load(file)

# Define a function to get a response for a given user input
def get_response(user_input, intents):
    user_input = user_input.lower()
    for intent, data in intents.items():
        for pattern in data["patterns"]:
            if user_input.find(pattern) != -1:
                return random.choice(data["responses"])
    return random.choice(intents["unknown"]["responses"])

# Define a function to interact with the chatbot
def chat_with_bot(intents):
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        response = get_response(user_input, intents)
        print("Chatbot:", response)

if __name__ == "__main__":
    chat_with_bot(intents)
