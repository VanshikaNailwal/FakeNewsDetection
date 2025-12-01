import streamlit as st
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import numpy as np
import pandas as pd
import math
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -------------------------
# Paths (use full absolute paths)
# -------------------------
MODEL_PATH = r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\logistic_regression.pkl"
VECTORIZER_PATH = r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\tfidf_vectorizer.pkl"

# -------------------------
# Utility: small cleaning function (must match training preprocessing)
# -------------------------
def clean_text_simple(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------
# Load model & vectorizer (with errors surfaced for debugging)
# -------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model from {MODEL_PATH}: {e}")
    st.stop()

try:
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    st.error(f"Could not load vectorizer from {VECTORIZER_PATH}: {e}")
    st.stop()

# helper for feature names compatibility
def get_feature_names(v):
    if hasattr(v, "get_feature_names_out"):
        return v.get_feature_names_out()
    elif hasattr(v, "get_feature_names"):
        return v.get_feature_names()
    else:
        return np.array([])

# -------------------------
# UI Theme + CSS
# -------------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.markdown("""
<style>
    .navbar {
        background-color: #11141A;
        padding: 20px;
        color: white;
        font-size: 28px;
        font-weight: 700;
        text-align:center;
        border-bottom: 2px solid #2E323C;
    }
    textarea {
        background-color: #181B27 !important;
        color: #E3E6F3 !important;
        border: 1px solid #2E6BFF !important;
        border-radius: 8px !important;
    }
    .stButton button {
        background-color: #2E6BFF !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
</style>
<div class="navbar">📰 Fake News Detection System</div>
""", unsafe_allow_html=True)

# -------------------------
# Classification Function (with input cleaning)
# -------------------------
def classify_text(text):
    text_clean = clean_text_simple(text)
    X = vectorizer.transform([text_clean])
    pred = model.predict(X)[0]

    # probability fallback (works for logistic / nb); decision_function fallback for SVM
    try:
        proba = model.predict_proba(X)[0]
        return int(pred), (float(proba[0]), float(proba[1]))
    except Exception:
        try:
            score = model.decision_function(X)[0]
            real = 1 / (1 + math.exp(-score))
            return int(pred), (1 - real, real)
        except Exception:
            # last fallback: use 0/1 deterministic
            return int(pred), (0.0, 1.0 if pred == 1 else 0.0)

# -------------------------
# Keyword Contributions (TF-IDF safe)
# -------------------------
def keyword_contributions(text, top_n=15):
    text_clean = clean_text_simple(text)
    fnames = get_feature_names(vectorizer)
    X = vectorizer.transform([text_clean]).toarray()[0]
    # ensure model.coef_ exists (linear models)
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["word","count","coef","contribution"]), pd.DataFrame(columns=["word","count","coef","contribution"])
    coefs = model.coef_[0]
    contrib = coefs * X
    idx = np.where(X > 0)[0]
    rows = []
    for i in idx:
        word = fnames[i] if i < len(fnames) else ""
        rows.append((word, float(X[i]), float(coefs[i]) if i < len(coefs) else 0.0, float(contrib[i]) if i < len(contrib) else 0.0))
    df = pd.DataFrame(rows, columns=["word","count","coef","contribution"])
    # remove super-common stopwords and tiny tokens
    df = df[~df["word"].isin(ENGLISH_STOP_WORDS)]
    df = df[df["word"].str.len() >= 3]
    pos = df.sort_values("contribution", ascending=False).head(top_n)
    neg = df.sort_values("contribution", ascending=True).head(top_n)
    return pos.reset_index(drop=True), neg.reset_index(drop=True)

# -------------------------
# Highlight Text (limit tokens)
# -------------------------
def highlight_text(text, df):
    tokens = df["word"].tolist()
    contrib = df.set_index("word")["contribution"].to_dict()
    result = text
    # sort tokens by length so longer multi-word tokens are replaced first
    for tok in sorted(tokens, key=len, reverse=True):
        if not tok:
            continue
        # word-boundary replacement; case-insensitive
        pattern = r"\b" + re.escape(tok) + r"\b"
        def repl(m):
            val = contrib.get(tok, 0.0)
            color = "#73A9FF" if val > 0 else "#FF6B6B"
            return f"<span style='background:{color}; padding:2px 6px; border-radius:4px'>{m.group(0)}</span>"
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return f"<div style='color:#E6E9F2; line-height:1.6; font-size:16px'>{result}</div>"

# -------------------------
# Gauge
# -------------------------
def create_gauge(real_score):
    fig, ax = plt.subplots(figsize=(3, 1.3))
    fig.patch.set_facecolor("#0D0F1A")
    ax.set_facecolor("#0D0F1A")
    ax.add_patch(Wedge((0,0), 1, 0, 180, facecolor="#2E323C"))
    ax.add_patch(Wedge((0,0), 1, 0, 180 * real_score, facecolor="#4D8BFF"))
    ax.add_patch(Circle((0,0), 0.6, facecolor="#0D0F1A"))
    ax.text(0, -0.12, f"{real_score*100:.1f}% Real", color="white",
            fontsize=11, fontweight="600", ha="center")
    ax.set_xlim(-1.05,1.05)
    ax.set_ylim(-0.05,1.05)
    ax.axis("off")
    plt.tight_layout()
    return fig

# -------------------------
# Sample News (Indian)
# -------------------------
SAMPLES_REAL = [
    "New Delhi: The Union Cabinet on Wednesday approved the construction of 1 crore affordable houses under the PM Awas Yojana (Urban) scheme.",
    "Mumbai: The Indian Space Research Organisation (ISRO) successfully launched the INSAT-3DS weather satellite from Sriharikota on Saturday."
]

SAMPLES_FAKE = [
    "WhatsApp message claims RBI will replace all ₹500 notes with a new purple version starting next week. No such announcement has been made.",
    "A viral post on social media claims that schools across India will remain closed for the next two months due to a 'severe solar storm'. This information is false."
]

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔎 Analyze", "🧠 Explain", "✨ Highlighted View", "🧸 ELI5 Mode", "ℹ️ About"
])

# ------------------------- ANALYZE -------------------------
with tab1:
    st.header("Enter or paste news text")
    col1, col2 = st.columns([3,1])
    with col1:
        user_input = st.text_area("", height=240, value=st.session_state.get("user_input", ""))
    with col2:
        st.write("Quick Samples:")
        if st.button("Load Real Sample"):
            st.session_state["user_input"] = SAMPLES_REAL[0]
            st.rerun()
        if st.button("Load Fake Sample"):
            st.session_state["user_input"] = SAMPLES_FAKE[0]
            st.rerun()

    if st.button("🚀 Classify News"):
        if not user_input.strip():
            st.warning("Enter text first.")
        else:
            pred, (fake_p, real_p) = classify_text(user_input)
            if pred == 1:
                st.success("This news appears to be REAL.")
            else:
                st.error("This news appears to be FAKE.")
            st.pyplot(create_gauge(real_p))
            st.metric("Fake Probability", f"{fake_p*100:.1f}%")
            st.metric("Real Probability", f"{real_p*100:.1f}%")
            st.session_state["text"] = user_input

# ------------------------- EXPLAIN -------------------------
with tab2:
    st.header("Why did the model predict this?")
    if "text" not in st.session_state:
        st.info("Classify something first.")
    else:
        pos, neg = keyword_contributions(st.session_state["text"])
        st.subheader("Words pushing REAL:")
        st.table(pos)
        st.subheader("Words pushing FAKE:")
        st.table(neg)

# ------------------------- HIGHLIGHTED VIEW -------------------------
with tab3:
    st.header("Highlighted Important Words")
    if "text" not in st.session_state:
        st.info("Run classification first.")
    else:
        pos, neg = keyword_contributions(st.session_state["text"], top_n=12)
        combined = pd.concat([pos, neg]).drop_duplicates("word")
        html = highlight_text(st.session_state["text"], combined)
        st.markdown(html, unsafe_allow_html=True)

# ------------------------- ELI5 MODE -------------------------
with tab4:
    st.header("Explain Like I'm 5 (ELI5 Mode)")
    if "text" not in st.session_state:
        st.info("Classify text first.")
    else:
        text = st.session_state["text"]
        pred, (fake_p, real_p) = classify_text(text)
        pos, neg = keyword_contributions(text)
        top_real = pos["word"].tolist()[:3]
        top_fake = neg["word"].tolist()[:3]
        if pred == 1:
            msg = f"""
            This news seems **REAL**.

            Imagine you're 5:

            - The story talks calmly and normally  
            - It uses grown-up words like **{", ".join(top_real)}**  
            - Nothing sounds too magical or impossible  

            So your computer friend thinks it is **true**.
            """
        else:
            msg = f"""
            This news seems **FAKE**.

            Imagine you're 5:

            - The story uses big dramatic words  
            - It says things that sound too crazy  
            - It uses words like **{", ".join(top_fake)}** that often appear in fake stories  

            So your computer friend thinks it is **made up**.
            """
        st.markdown(f"<div style='color:#E6E9F2; font-size:18px'>{msg}</div>", unsafe_allow_html=True)

# ------------------------- ABOUT -------------------------
with tab5:
    st.header("About this App")
    st.write("""
    This is a Fake News Detector built using:
    - Logistic Regression  
    - TF-IDF Vectorizer (with unigrams + bigrams)  
    - Keyword Contribution Explainability  
    - Highlighted Text Inspection  
    - ELI5 Simple Explanation Mode  

    The goal is to make fake news detection understandable and user-friendly.
    """)

