from flask import Flask, request, render_template
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import os
import pickle

app = Flask(__name__)

# Define global variables for model and tokenizer
model = None
tokenizer = None

def load_tokenizer():
    global tokenizer
    tokenizer_path = "C:/major_project_interface/tokenizer.pickle"

    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Function to generate summary
def generate_summary(input_text):
    global tokenizer, model
    
    if tokenizer is None:
        load_tokenizer()
    
    # Preprocess input text
    preprocessed_input = preprocess_text(input_text)
    
    # Tokenize and pad preprocessed input text
    input_sequence = tokenizer.texts_to_sequences([preprocessed_input])
    input_padded = pad_sequences(input_sequence, maxlen=100, padding='post')
    
    # Generate summary using the pre-trained model
    predicted_sequence = model.predict(input_padded)
    
    # Decode the predicted sequence
    decoded_summary = []
    for seq in predicted_sequence[0]:
        sampled_token_index = np.argmax(seq)
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')
        if sampled_word:
            decoded_summary.append(sampled_word)
    
    return ' '.join(decoded_summary)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    input_text = request.form['input_text']
    
    if len(input_text.strip()) == 0:
        return render_template('output.html', input_text=input_text, predicted_summary="Input text is empty or invalid.")
    
    # Generate summary for the input text
    predicted_summary = generate_summary(input_text)
    
    return render_template('output.html', input_text=input_text, predicted_summary=predicted_summary)

if __name__ == '__main__':
    model_path = os.path.abspath("lstm_model.h5")
    model = load_model(model_path)
    app.run(debug=True)
