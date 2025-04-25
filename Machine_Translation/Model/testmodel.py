import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects

# Load dataset
file_path = r"C:\Users\mithi\Desktop\MT\Book1.xlsx"
df = pd.read_excel(file_path)

# Assume first column is English, second column is Bengali
english_texts = df.iloc[:, 0].astype(str).tolist()
bengali_texts = df.iloc[:, 1].astype(str).tolist()

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Zঀ-৿ ]", "", text)  # Retain English & Bengali letters
    return text.strip()

# Apply cleaning
english_texts = [clean_text(sent) for sent in english_texts]
bengali_texts = [clean_text(sent) for sent in bengali_texts]

# Add start & end tokens for Bengali sentences
bengali_texts = ["<start> " + sent + " <end>" for sent in bengali_texts]

# Splitting Data (2/3 Training, 1/3 Testing)
_, test_english, _, test_bengali = train_test_split(
    english_texts, bengali_texts, test_size=1/3, random_state=42
)

# Register custom objects if required
def NotEqual(x, y):
    return tf.not_equal(x, y)

get_custom_objects()["NotEqual"] = NotEqual

# Load trained model
model_path = r"C:\Users\mithi\Desktop\MT\Model\seq2seq_translation_model.h5"
model = load_model(model_path, custom_objects={"NotEqual": NotEqual})

# Tokenization
eng_tokenizer = Tokenizer(filters="")
eng_tokenizer.fit_on_texts(english_texts)
eng_vocab_size = len(eng_tokenizer.word_index) + 1

ben_tokenizer = Tokenizer(filters="")
ben_tokenizer.fit_on_texts(bengali_texts)
ben_vocab_size = len(ben_tokenizer.word_index) + 1

# Convert text to sequences
test_english_seq = eng_tokenizer.texts_to_sequences(test_english)

# Padding
max_eng_len = max(len(seq) for seq in test_english_seq)
max_ben_len = max(len(seq) for seq in ben_tokenizer.texts_to_sequences(bengali_texts))
test_english_seq = pad_sequences(test_english_seq, maxlen=max_eng_len, padding="post")

# Function to translate test sentences
def translate_sentence(sentence):
    sequence = eng_tokenizer.texts_to_sequences([sentence])
    sequence = pad_sequences(sequence, maxlen=max_eng_len, padding="post")
    prediction = model.predict([sequence, np.zeros((1, max_ben_len))])
    predicted_indices = np.argmax(prediction, axis=-1)[0]
    translated_words = [ben_tokenizer.index_word.get(idx, "") for idx in predicted_indices if idx > 0]
    return " ".join(translated_words).replace("<start>", "").replace("<end>", "").strip()

# Testing on 1/3rd of data
for i in range(5):  # Print first 5 translations
    print(f"English: {test_english[i]}")
    print(f"Predicted Bengali: {translate_sentence(test_english[i])}")
    print(f"Actual Bengali: {test_bengali[i]}\n")
