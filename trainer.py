import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample conversation data
conversations = [
    ("Hello", "Hi! How can I help you?"),
    ("What's the weather like today?", "The weather is sunny with a high of 28°C."),
    # Add more conversation pairs here
]

# Data preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts([message for _, message in conversations])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
output_sequences = []

for input_text, output_text in conversations:
    input_seq = tokenizer.texts_to_sequences([input_text])[0]
    output_seq = tokenizer.texts_to_sequences([output_text])[0]

    input_sequences.append(input_seq)
    output_sequences.append(output_seq)

max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post')

# Model architecture
model = keras.Sequential([
    Embedding(total_words, 64, input_length=max_sequence_length),
    LSTM(128),
    Dense(total_words, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training loop
epochs = 100
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.fit(input_sequences, np.expand_dims(output_sequences, axis=-1), verbose=1)

# Real-time conversation interaction
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    user_input_seq = tokenizer.texts_to_sequences([user_input])[0]
    user_input_seq = pad_sequences([user_input_seq], maxlen=max_sequence_length, padding='post')

    response_seq = model.predict(user_input_seq)
    response_text = tokenizer.sequences_to_texts(response_seq)[0]
    print(f"Bot: {response_text}")
