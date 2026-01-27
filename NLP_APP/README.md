# 🧠 NLP Preprocessing Streamlit App

An interactive **Natural Language Processing (NLP) web application** built using **Streamlit**, designed to demonstrate essential NLP preprocessing techniques in a clear, visual, and beginner-friendly way.

🔗 **Live App**: https://nlp-app-ibb.streamlit.app/

---

## 📌 Project Motivation

Natural Language Processing is a core part of modern AI systems such as **chatbots, search engines, recommendation systems, and sentiment analysis tools**.  
However, raw text data cannot be directly used by machine learning models.

👉 This application is built to **bridge the gap between raw text and machine-ready data** by demonstrating the most important NLP preprocessing steps interactively.

The app helps students and beginners:
- Understand how text is transformed step by step
- Visually compare different NLP techniques
- Learn why preprocessing is crucial before model training

---

## 🎯 Why This App and These Techniques?

- **Tokenization** helps break text into meaningful units
- **Text Cleaning** removes noise and irrelevant information
- **Stemming & Lemmatization** normalize words
- **Bag of Words & TF-IDF** convert text into numerical features
- **Word Embeddings** capture semantic meaning

📌 These techniques form the **foundation of almost every NLP pipeline**, making this app useful for:
- Academic learning
- NLP labs
---

## 🛠️ Technologies Used

- **Python**
- **Streamlit** – Interactive web interface
- **NLTK** – Tokenization and stemming
- **spaCy** – Lemmatization and embeddings
- **Scikit-learn** – Bag of Words and TF-IDF
- **Pandas & Matplotlib** – Data handling and visualization

---

## ✍️ Input

- User provides English text as input
- One NLP technique can be selected at a time
- Processed output is shown instantly

---

## 🧩 NLP Techniques Implemented

### 1️⃣ Tokenization
Splits text into:
- Sentences
- Words
- Characters  

📌 Helps understand text structure.

---

### 2️⃣ Text Cleaning
- Converts text to lowercase
- Removes punctuation and numbers
- Removes stopwords  

📌 Improves data quality by removing noise.

---

### 3️⃣ Stemming
Reduces words to their root form using:
- Porter Stemmer
- Lancaster Stemmer  

Example:  
`playing → play`

📌 Fast but may reduce accuracy.

---

### 4️⃣ Lemmatization
Converts words to meaningful base forms using grammar.

Example:  
`better → good`

📌 More accurate and linguistically correct.

---

### 5️⃣ Bag of Words (BoW)
- Represents text using word frequency
- Displays frequency table
- Visualized using a pie chart (Top words)

📌 Simple numerical representation of text.

---

### 6️⃣ TF-IDF
- Assigns importance scores to words
- Reduces impact of common words
- Displayed using a bar chart

📌 Highlights important terms in text.

---

### 7️⃣ Word Embeddings
- Converts words into numerical vectors
- Displays word-level and sentence-level embeddings

📌 Captures semantic meaning and context.

---

## 📊 Output Representation

- Interactive tables
- Graphical visualizations
- Clean and user-friendly layout

---

## 🚀 Future Scope

- File upload support (TXT / PDF)
- N-grams
- Cosine similarity
- Named Entity Recognition (NER)
- Advanced embeddings (Word2Vec, GloVe)

---

## 👨‍💻 Author

**Indranil Barua Betal**  
Computer Science & Engineering Student  
Haldia Institute of Technology  

📧 Email: indranilbaruabetal@gmail.com  
