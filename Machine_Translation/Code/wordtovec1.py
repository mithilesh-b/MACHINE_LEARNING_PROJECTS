# Import necessary libraries
from tensorflow.keras.preprocessing.text import one_hot # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, Flatten, Dense # type: ignore

# Sample data
sentences = [
    'গ্রামীণ এলাকায় খায়নের জলের প্রাইপ্যতা উন্নত করার লিগ্গা সর্বাত্মক প্রচেষ্টা চালাতে হইবো',
    'জৈবপ্রযুক্তির ক্ষেত্রে, জৈব সার, জলজ চাষ, বায়োমাস উৎপাদন সংক্রান্ত উন্নয়ন',
    'পুরুষরা সাতটা গাছের চারপাশে জলের ছিটা দিতাছে',
    'জল দূষণ সামুদ্রিক বাস্তুতন্ত্র আর জীবনরে বিপন্ন করে'
]

# Vocabulary size (arbitrarily set; should be larger than total unique words in the corpus)
vocab_size = 50

# Encoding the sentences into integer sequences using one_hot
encoded_sentences = [one_hot(sentence, vocab_size) for sentence in sentences]
print("Encoded Sentences:", encoded_sentences)

# Padding sequences to ensure uniform length
max_length = max(len(seq) for seq in encoded_sentences)  # Find the longest sentence length
padded_sentences = pad_sequences(encoded_sentences, maxlen=max_length, padding='post')
print("Padded Sentences:\n", padded_sentences)

# embedding dimension
embedding_dim = 12

# model with an embedding layer
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(Flatten())  
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Run a single prediction to initialize the embeddings
model.predict(padded_sentences)

# Get the embedding weights
embeddings = model.layers[0].get_weights()[0]
print("Embedding Matrix:\n", embeddings)
