import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple feedforward neural network
class ChatbotNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Load intents data
intents = json.loads(open('intents.json').read())

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the PyTorch model
input_size = len(words)
hidden_size = 8
output_size = len(classes)
model = ChatbotNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('chatbot_model.pkl'))
model.eval()

lemmatizer = WordNetLemmatizer()


def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    with torch.no_grad():
        input_data = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)
        outputs = model(input_data)
    predicted_class_index = torch.argmax(outputs, dim=1).item()
    predicted_class = classes[predicted_class_index]
    return predicted_class


print('Chatbot: Hello! How can I assist you today?')
while True:
    user_input = input('You: ')

    if user_input.lower() == 'exit':
        print('Chatbot: Goodbye!')
        break

    predicted_intent = predict_class(user_input)

    for intent_data in intents['intents']:
        if intent_data['tag'] == predicted_intent:
            responses = intent_data['responses']
            print('Chatbot:', random.choice(responses))