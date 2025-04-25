import pandas as pd
from sklearn.metrics import cohen_kappa_score

def preprocess_text(text):
    """Normalize text by converting to lowercase and stripping spaces."""
    if isinstance(text, str):
        return text.strip().lower()
    return ""

def compute_cohens_kappa(file1, file2, sheet_name=0, eng_col="English", ben_col="Bangal"):
    # Load the Excel files (first sheet by default)
    df1 = pd.read_excel(file1, sheet_name="Sheet1")
    df2 = pd.read_excel(file2, sheet_name="Sheet1")

    # Check if the necessary columns exist
    for col in [eng_col, ben_col]:
        if col not in df1.columns or col not in df2.columns:
            raise ValueError(f"Column '{col}' not found in one of the provided files")

    # Extract Bengali translations for comparison
    translations1 = df1[ben_col].dropna().apply(preprocess_text).tolist()
    translations2 = df2[ben_col].dropna().apply(preprocess_text).tolist()

    # Ensure both lists are of the same length
    min_length = min(len(translations1), len(translations2))
    translations1 = translations1[:min_length]
    translations2 = translations2[:min_length]

    # Compute Cohen's Kappa score
    kappa_score = cohen_kappa_score(translations1, translations2)
    return kappa_score

# File paths
file1 = r"C:\Users\mithi\Desktop\MT\Book1.xlsx"
file2 = r"C:\Users\mithi\Desktop\MT\MachineTranslation.xlsx"

# Compute Cohen's Kappa
kappa = compute_cohens_kappa(file1, file2, eng_col="English", ben_col="Bangal")
print(f"Cohen's Kappa Score: {kappa}")
