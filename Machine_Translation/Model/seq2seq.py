import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Load dataset
file_path = r"C:\Users\mithi\Desktop\MT\MachineTranslation.xlsx"
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
train_english, test_english, train_bengali, test_bengali = train_test_split(
    english_texts, bengali_texts, test_size=1/3, random_state=42
)

# Tokenization
eng_tokenizer = Tokenizer(filters="")
eng_tokenizer.fit_on_texts(train_english)
eng_vocab_size = len(eng_tokenizer.word_index) + 1

ben_tokenizer = Tokenizer(filters="")
ben_tokenizer.fit_on_texts(train_bengali)
ben_vocab_size = len(ben_tokenizer.word_index) + 1

# Convert text to sequences
train_english_seq = eng_tokenizer.texts_to_sequences(train_english)
train_bengali_seq = ben_tokenizer.texts_to_sequences(train_bengali)

# Padding
max_eng_len = max(len(seq) for seq in train_english_seq)
max_ben_len = max(len(seq) for seq in train_bengali_seq)

train_english_seq = pad_sequences(train_english_seq, maxlen=max_eng_len, padding="post")
train_bengali_seq = pad_sequences(train_bengali_seq, maxlen=max_ben_len, padding="post")

# Model Parameters
embedding_dim = 256
lstm_units = 512

# Encoder
encoder_inputs = Input(shape=(max_eng_len,))
enc_embedding = Embedding(eng_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(lstm_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_embedding)

# Decoder
decoder_inputs = Input(shape=(max_ben_len,))
dec_embedding = Embedding(ben_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_embedding, initial_state=[state_h, state_c])
decoder_dense = Dense(ben_vocab_size, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()

# Prepare Decoder Target Data
decoder_target_data = np.zeros_like(train_bengali_seq)
decoder_target_data[:, :-1] = train_bengali_seq[:, 1:]

# Train the model
model.fit(
    [train_english_seq, train_bengali_seq],
    decoder_target_data,
    batch_size=64,
    epochs=50,
    validation_split=0.2
)

# Save the model
model.save("seq2seq_translation_model.h5")
