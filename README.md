📰 News Article Classification (Fake vs Real)

📌 Overview
This project is a Fake News Detection App that classifies news headlines or full articles as Fake or Real using Natural Language Processing (NLP) and Machine Learning techniques.  
It helps identify misleading information by analyzing text patterns, keywords, and linguistic cues.

---

🎯 Objectives
- Detect whether a given news article is fake or real.
- Build a robust NLP pipeline for text preprocessing and feature extraction.
- Train and evaluate a machine learning model for accurate classification.
- Provide a Streamlit web app for real-time fake news detection.

---

⚙️ Tech Stack
- Python
- Pandas, NumPy
- NLTK (Natural Language Toolkit)
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

---

🧠 Model Workflow
1. Dataset Collection
   Used labeled fake and real news datasets (`Fake.csv`, `True.csv`) from Kaggle.  
   [Download Dataset from Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

2. Data Cleaning & Preprocessing  
   - Lowercasing  
   - Stopword removal  
   - Tokenization  
   - Lemmatization  

3. Feature Extraction 
   - Text transformed using TF-IDF Vectorization  

4. Model Training
   - Classifier: Logistic Regression  
   - Evaluated with Accuracy, F1-score, and Confusion Matrix  

5. Streamlit Deployment
   - Interactive UI for entering headlines or full articles  
   - Displays prediction with confidence score and contributing keywords  

---

📊 Model Performance
| Metric | Score |
|:-------|:------:|
| Accuracy | 0.40 |
| F1-Score | 0.57 |

(Based on results from the trained model.)

---

🖥️ How to Run Locally
```bash
# Clone the repository
git clone https://github.com/shibikasrig/News-Article-Classification-Fake-Real-.git
cd News-Article-Classification-Fake-Real-

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

📁 fake-news-classification
│
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── fake_news_model.pkl     # Trained ML model
├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
├── Fake.csv / True.csv     # Dataset files (from Kaggle)
├── requirements.txt        # Python dependencies
└── report.txt              # Project report summary

