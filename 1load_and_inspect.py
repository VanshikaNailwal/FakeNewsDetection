import pandas as pd
from sklearn.model_selection import train_test_split

print("\n--- Loading Data ---")

# -----------------------------
# Load datasets
# -----------------------------
fake_df = pd.read_csv(r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\Fake.csv")
true_df = pd.read_csv(r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\True.csv")
other_df = pd.read_csv(r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\news_dataset.csv")

# -----------------------------
# Print shapes
# -----------------------------
print(f"\nFake.csv shape        : {fake_df.shape}")
print(f"True.csv shape        : {true_df.shape}")
print(f"news_dataset.csv shape: {other_df.shape}")

# -----------------------------
# Add labels for True/Fake datasets
# -----------------------------
fake_df["label"] = 0   # Fake
true_df["label"] = 1   # True

# -----------------------------
# Standardize third dataset labels
# -----------------------------
other_df["label"] = other_df["label"].map({"FAKE": 0, "REAL": 1})

# -----------------------------
# Standardize columns (keep only text + label)
# -----------------------------
fake_df = fake_df.rename(columns={"text": "clean_text"})
true_df = true_df.rename(columns={"text": "clean_text"})
other_df = other_df.rename(columns={"text": "clean_text"})

fake_df = fake_df[["clean_text", "label"]]
true_df = true_df[["clean_text", "label"]]
other_df = other_df[["clean_text", "label"]]

# -----------------------------
# Combine all 3 datasets
# -----------------------------
data = pd.concat([fake_df, true_df, other_df], ignore_index=True)

# Shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------
# Show final info
# -----------------------------
print("\n--- Final Combined Dataset ---")
print(f"Total rows : {data.shape[0]}")
print(f"Total cols : {data.shape[1]}")

print("\n--- Sample Rows ---")
print(data.head())
