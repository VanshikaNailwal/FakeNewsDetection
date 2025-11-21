import streamlit as st
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import numpy as np
import pandas as pd
import math
import re
import time

# -------------------------
# Paths
# -------------------------
MODEL_PATH = r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\logistic_regression.pkl"
VECTORIZER_PATH = r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\bow_vectorizer.pkl"

# -------------------------
# Load model & vectorizer
# -------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

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
# Classification Function
# -------------------------
def classify_text(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    try:
        proba = model.predict_proba(X)[0]
        return pred, (proba[0], proba[1])
    except:
        score = model.decision_function(X)[0]
        real = 1 / (1 + math.exp(-score))
        return pred, (1-real, real)

# -------------------------
# Keyword Contributions
# -------------------------
def keyword_contributions(text):
    fnames = vectorizer.get_feature_names_out()
    X = vectorizer.transform([text]).toarray()[0]
    coefs = model.coef_[0]
    contrib = coefs * X
    idx = np.where(X > 0)[0]

    rows = [(fnames[i], X[i], coefs[i], contrib[i]) for i in idx]
    df = pd.DataFrame(rows, columns=["word","count","coef","contribution"])

    pos = df.sort_values("contribution", ascending=False).head(10)
    neg = df.sort_values("contribution", ascending=True).head(10)
    return pos, neg

# -------------------------
# Highlight Text
# -------------------------
def highlight_text(text, df):
    tokens = df["word"].tolist()
    contrib = df.set_index("word")["contribution"].to_dict()

    result = text
    for tok in sorted(tokens, key=len, reverse=True):
        pattern = r"\b" + re.escape(tok) + r"\b"
        def repl(m):
            val = contrib.get(tok, 0)
            color = "#73A9FF" if val > 0 else "#FF6B6B"
            return f"<span style='background:{color}; padding:2px 6px; border-radius:4px'>{m.group(0)}</span>"
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)

    return f"<div style='color:#E6E9F2; line-height:1.6; font-size:16px'>{result}</div>"

# -------------------------
# Smaller Gauge Meter
# -------------------------
def create_gauge(real_score):
    fig, ax = plt.subplots(figsize=(3, 1.3))  # SMALLER SIZE
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
# Sample News
# -------------------------
SAMPLES_REAL = [
    "WASHINGTON (Reuters) - The United States on Thursday approved a new $425 million aid package for Ukraine.",
]

SAMPLES_FAKE = [
    "NASA has confirmed Earth will be dark for six days next month due to planetary alignment.",
]

# -------------------------
# TABS (without similarity tab)
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔎 Analyze", "🧠 Explain", "✨ Highlighted View", "🧸 ELI5 Mode", "ℹ️ About"
])

# ------------------------- ANALYZE -------------------------
with tab1:
    st.header("Enter or paste news text")

    col1, col2 = st.columns([3,1])

    with col1:
        user_input = st.text_area(
            "",
            height=240,
            value=st.session_state.get("user_input", "")
        )

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
        pos, neg = keyword_contributions(st.session_state["text"])
        combined = pd.concat([pos, neg])
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
    - Bag-of-Words Vectorizer  
    - Keyword Contribution Explainability  
    - Highlighted Text Inspection  
    - ELI5 Simple Explanation Mode  

    The goal is to make fake news detection understandable and user-friendly.
    """)

