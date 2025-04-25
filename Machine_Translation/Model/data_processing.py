import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

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

# Print sample sizes
print(f"Training size: {len(train_english)}, Testing size: {len(test_english)}")
