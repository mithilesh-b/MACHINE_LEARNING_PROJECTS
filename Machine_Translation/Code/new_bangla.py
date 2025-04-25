import gensim
import pandas as pd 
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')


file_path = '.\MachineTranslation.xlsx'
df = pd.read_excel(file_path)
Bangal_column = df['Bangal']
print(Bangal_column.head())
sentences = df['Bangal'].dropna().tolist()
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
w2v_model = Word2Vec( sentences=tokenized_sentences,vector_size=50,window=5,min_count=2,workers=6)
w2v_model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=100)

w2v_model.save("./new_word2vec-bangal.model")

#similar_words = w2v_model.wv.most_similar('আমি', topn = 2)
#print(similar_words)
word1 = "আমি"
word2 = "আমার"

if word1 in w2v_model.wv and word2 in w2v_model.wv:
    similarity = w2v_model.wv.similarity(w1=word1, w2=word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity}")
else:
    print(f"One or both words '{word1}' and '{word2}' are not in the vocabulary.")

