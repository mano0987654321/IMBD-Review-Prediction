## Step 1: Import all the necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Set vocabulary size used in training
VOCAB_SIZE = 10000

# Load the dataset's word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pretrained model
model = load_model('simple_rnn_imdb.h5')

# Step-2: Helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input safely
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []

    for word in words:
        idx = word_index.get(word, 2) + 3  # +3 for reserved tokens
        if idx < VOCAB_SIZE:  # only include if within known range
            encoded_review.append(idx)
        else:
            encoded_review.append(2)  # use OOV token for unknown words

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Streamlit App
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below and classify it as **positive** or **negative**.")

# User Input
user_input = st.text_area('📝 Movie Review')

if st.button('🎯 Classify'):
    if user_input.strip():
        preprocess_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocess_input)
        sentiment = 'Positive 😊' if prediction[0][0] > 0.5 else 'Negative 😞'

        # Display result
        st.success(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]:.4f}')
    else:
        st.warning("Please enter a review before clicking 'Classify'.")
else:
    st.info('Awaiting your input above.')
