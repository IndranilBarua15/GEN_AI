import streamlit as st
import nltk
import spacy
import string
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# -------------------- DOWNLOAD NLTK DATA --------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# -------------------- LOAD SPACY MODEL --------------------
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    st.stop()

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(
    page_title="NLP Preprocessing App",
    layout="wide"
)

st.title("🧠 NLP Preprocessing App")
st.write("Tokenization, Cleaning, Stemming, Lemmatization, BoW, TF-IDF & Word Embeddings")

# -------------------- USER INPUT --------------------
text = st.text_area(
    "Enter text for NLP processing",
    height=150,
    placeholder="Example: Aman is the HOD of HIT and loves NLP"
)

# -------------------- SIDEBAR --------------------
option = st.sidebar.radio(
    "Select NLP Technique",
    (
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "Bag of Words",
        "TF-IDF",
        "Word Embeddings"
    )
)

# -------------------- PROCESS BUTTON --------------------
if st.button("Process Text"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    # ---------------- TOKENIZATION ----------------
    if option == "Tokenization":
        st.subheader("🔹 Tokenization Output")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Sentence Tokenization")
            st.write(sent_tokenize(text))

        with col2:
            st.markdown("### Word Tokenization")
            st.write(word_tokenize(text))

        with col3:
            st.markdown("### Character Tokenization")
            st.write(list(text))

    # ---------------- TEXT CLEANING ----------------
    elif option == "Text Cleaning":
        st.subheader("🔹 Text Cleaning Output")

        text_lower = text.lower()
        cleaned_text = "".join(
            ch for ch in text_lower
            if ch not in string.punctuation and not ch.isdigit()
        )

        doc = nlp(cleaned_text)
        final_words = [
            token.text for token in doc
            if not token.is_stop and token.text.strip()
        ]

        st.markdown("### Original Text")
        st.write(text)

        st.markdown("### Cleaned Text")
        st.write(" ".join(final_words))

    # ---------------- STEMMING ----------------
    elif option == "Stemming":
        st.subheader("🔹 Stemming Output")

        words = word_tokenize(text)

        porter = PorterStemmer()
        lancaster = LancasterStemmer()

        df = pd.DataFrame({
            "Original Word": words,
            "Porter Stemmer": [porter.stem(w) for w in words],
            "Lancaster Stemmer": [lancaster.stem(w) for w in words]
        })

        st.dataframe(df, use_container_width=True)

    # ---------------- LEMMATIZATION ----------------
    elif option == "Lemmatization":
        st.subheader("🔹 Lemmatization using spaCy")

        doc = nlp(text)
        df = pd.DataFrame(
            [(token.text, token.pos_, token.lemma_) for token in doc],
            columns=["Word", "POS", "Lemma"]
        )

        st.dataframe(df, use_container_width=True)

    # ---------------- BAG OF WORDS ----------------
    elif option == "Bag of Words":
        st.subheader("🔹 Bag of Words Representation")

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([text])

        df = pd.DataFrame({
            "Word": vectorizer.get_feature_names_out(),
            "Frequency": X.toarray()[0]
        }).sort_values(by="Frequency", ascending=False)

        st.dataframe(df, use_container_width=True)

        st.markdown("### Word Frequency Distribution (Top 10)")
        df_top = df.head(10)

        fig, ax = plt.subplots()
        ax.pie(
            df_top["Frequency"],
            labels=df_top["Word"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

    # ---------------- TF-IDF ----------------
    elif option == "TF-IDF":
        st.subheader("🔹 TF-IDF Representation")

        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform([text])

        df = pd.DataFrame({
            "Word": tfidf.get_feature_names_out(),
            "TF-IDF Score": X.toarray()[0]
        }).sort_values(by="TF-IDF Score", ascending=False)

        st.dataframe(df, use_container_width=True)

        st.markdown("### TF-IDF Distribution (Top 10)")
        df_top = df.head(10)

        fig, ax = plt.subplots()
        ax.bar(df_top["Word"], df_top["TF-IDF Score"])
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ---------------- WORD EMBEDDINGS ----------------
    elif option == "Word Embeddings":
        st.subheader("🔹 Word Embeddings (spaCy)")

        doc = nlp(text)

        data = []
        for token in doc:
            if token.is_alpha and not token.is_stop and token.has_vector:
                data.append({
                    "Word": token.text,
                    "Vector Size": token.vector.shape[0],
                    "Vector (First 5 values)": token.vector[:5]
                })

        if not data:
            st.warning("No embeddings found. Use meaningful English text.")
            st.stop()

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

        st.markdown("### Sentence Embedding")
        sentence_vector = doc.vector
        st.write(f"Vector Size: {sentence_vector.shape[0]}")
        st.write("First 10 values:")
        st.write(sentence_vector[:10])
