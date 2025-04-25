import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not available
nltk.download('stopwords')
nltk.download('punkt')

# Define Bengali and English stopwords
bengali_stopwords = set(["আর", "কিন্তু", "তাইলে", "কেরে", "কিতা", "যেই", "এর লিজ্ঞা", "তুমি", "আমি", "হে", "এইডা", "ওইডা", "এইডা", "ওইডা", "তারা", "তোমার", "আমার", "আপনের", "আপনে", "কেরে", "হইছে"])
english_stopwords = set(stopwords.words('english'))

# Function to process text
def process_text(text, lang="bengali"):
    if isinstance(text, str):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = nltk.word_tokenize(text)
        stop_words = bengali_stopwords if lang == "bengali" else english_stopwords
        return [word for word in words if word not in stop_words]
    return []

# Function to get unique words and frequencies
def get_word_frequencies(file_path, sheet_name=0, eng_col="English", ben_col="Bangal"):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    if eng_col not in df.columns or ben_col not in df.columns:
        raise ValueError("Column names do not match the dataset.")
    
    # Process English and Bengali text
    english_words = []
    bengali_words = []
    
    for text in df[eng_col].dropna():
        english_words.extend(process_text(text, lang="english"))
    
    for text in df[ben_col].dropna():
        bengali_words.extend(process_text(text, lang="bengali"))
    
    # Get word frequencies
    english_freq = Counter(english_words)
    bengali_freq = Counter(bengali_words)
    
    return english_freq, bengali_freq

# File paths
file1 = r"C:\Users\mithi\Desktop\MT\Book1.xlsx"
file2 = r"C:\Users\mithi\Desktop\MT\MachineTranslation.xlsx"

# Compute word frequencies for both files
eng_freq1, ben_freq1 = get_word_frequencies(file1)
eng_freq2, ben_freq2 = get_word_frequencies(file2)

# Convert to DataFrame for exporting
eng_df1 = pd.DataFrame(eng_freq1.items(), columns=["Word", "Frequency (File1)"])
ben_df1 = pd.DataFrame(ben_freq1.items(), columns=["Word", "Frequency (File1)"])
eng_df2 = pd.DataFrame(eng_freq2.items(), columns=["Word", "Frequency (File2)"])
ben_df2 = pd.DataFrame(ben_freq2.items(), columns=["Word", "Frequency (File2)"])

# Merge data
eng_df = pd.merge(eng_df1, eng_df2, on="Word", how="outer").fillna(0)
ben_df = pd.merge(ben_df1, ben_df2, on="Word", how="outer").fillna(0)

# Save results to Excel
with pd.ExcelWriter("word_frequencies.xlsx") as writer:
    eng_df.to_excel(writer, sheet_name="English", index=False)
    ben_df.to_excel(writer, sheet_name="Bengali", index=False)

print("Word frequencies saved to 'word_frequencies.xlsx'")
