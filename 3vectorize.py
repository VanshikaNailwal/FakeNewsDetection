import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("\n📂 Loading cleaned dataset...")
df = pd.read_csv(r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\clean_news.csv")
print(f"Dataset shape: {df.shape}")

# Safety cleaning
df = df[df["clean_text"].str.strip() != ""]
df = df.dropna(subset=["clean_text"])

# Features and labels
X = df["clean_text"].astype(str)
y = df["label"]

print("\n🔀 Splitting into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)

print("\n📝 Applying TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),      # NEW: Use bigrams for better contextual understanding
    stop_words='english'    # NEW: Cleaner signal
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\n📊 VECTOR INFO:")
print("X_train shape:", X_train_vec.shape)
print("X_test shape:", X_test_vec.shape)
print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

print("\n💾 Saving TF-IDF data...")
joblib.dump(X_train_vec, r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\X_train.pkl")
joblib.dump(X_test_vec, r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\X_test.pkl")
joblib.dump(y_train, r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\y_train.pkl")
joblib.dump(y_test, r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\y_test.pkl")
joblib.dump(vectorizer, r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\tfidf_vectorizer.pkl")

print("\n✅ TF-IDF Vectorization complete!")
