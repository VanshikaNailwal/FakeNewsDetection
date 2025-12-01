import pandas as pd
import re
import string

# ------------------------------------------------------
# CLEANING FUNCTION
# ------------------------------------------------------
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

# ------------------------------------------------------
# MAIN PREPROCESSING SCRIPT
# ------------------------------------------------------
print("\n Loading datasets...")

# Load original True/Fake datasets
true_df = pd.read_csv(r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\True.csv")
fake_df = pd.read_csv(r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\Fake.csv")

# Load the 3rd dataset
other_df = pd.read_csv(r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\news_dataset.csv")

print(" True:", true_df.shape)
print(" Fake:", fake_df.shape)
print(" News_dataset:", other_df.shape)

# ------------------------------------------------------
# Add labels for True/Fake
# ------------------------------------------------------
true_df["label"] = 1
fake_df["label"] = 0

# ------------------------------------------------------
# Process 3rd dataset labels (REAL/FAKE → 1/0)
# ------------------------------------------------------
other_df["label"] = other_df["label"].map({"REAL": 1, "FAKE": 0})

# ------------------------------------------------------
# Standardize column names
# ------------------------------------------------------
true_df["clean_text"] = (true_df["title"].astype(str) + " " + true_df["text"].astype(str))
fake_df["clean_text"] = (fake_df["title"].astype(str) + " " + fake_df["text"].astype(str))
other_df = other_df.rename(columns={"text": "clean_text"})

# Keep only clean_text + label
true_df = true_df[["clean_text", "label"]]
fake_df = fake_df[["clean_text", "label"]]
other_df = other_df[["clean_text", "label"]]

# ------------------------------------------------------
# Combine all datasets
# ------------------------------------------------------
df = pd.concat([true_df, fake_df, other_df], ignore_index=True)
print("\n Combined shape:", df.shape)

# ------------------------------------------------------
# Missing values
# ------------------------------------------------------
print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum())

# ------------------------------------------------------
# Clean the text
# ------------------------------------------------------
print("\n Cleaning text...")
df["clean_text"] = df["clean_text"].astype(str).apply(clean_text)

# ------------------------------------------------------
# REMOVE MISSING ROWS (Option A)
# ------------------------------------------------------
print("\n Removing missing values...")

df = df.dropna(subset=["clean_text"])  # drop NaN
df = df[df["clean_text"].str.strip() != ""]  # drop empty strings

print(" New shape after removing missing rows:", df.shape)

# ------------------------------------------------------
# Save final cleaned dataset
# ------------------------------------------------------
output_path = r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\clean_news.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n Saved cleaned dataset to: {output_path}")

# Load again to confirm
df = pd.read_csv(output_path)

print("\n--- Sample Cleaned Row ---")
print(df.iloc[0])
