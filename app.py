# app.py
import streamlit as st
import joblib
from fpdf import FPDF
import io
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection App")
st.write("Enter a news headline or full article to check if it's Fake or Real")

# Input type
input_type = st.radio("Choose Input Type:", ["Headline Only", "Full Article"])

# Sample news
sample_news = {
    "Teleportation device sold online": 
    "A viral post claims that a teleportation machine is being sold online for $999, "
    "but experts confirm this is completely false and a marketing gimmick.",
    
    "Chocolate increases lifespan by 50 years": 
    "False claims about chocolate consumption making humans live 150 years.",
    
    "Aliens invade New York City": 
    "Reports claim that aliens landed in New York City last night, causing panic among residents. Scientists confirm this is false.",
    
    "Cure for COVID-19 found in household spices": 
    "A viral post claims that turmeric and garlic can cure COVID-19, but health authorities deny this.",
    
    "Politician secretly controls the moon": 
    "A fake report circulates that a famous politician has a secret moon base, spreading conspiracy theories.",
    
    "Time travel device invented in garage": 
    "Social media spreads news about a man claiming he built a time machine, but it's completely fake.",
    
    "Chocolate cures all diseases": 
    "A post claims eating chocolate daily cures all illnesses; medical experts confirm it's false.",
    
    "Man grows wings overnight": 
    "Viral video shows a man growing wings overnight; biologists confirm it is fabricated.",
    
    "Water turns into gold with new device": 
    "A fake news article claims a device can turn water into gold; scientists debunk it.",
    
    "Moon landing staged in Hollywood": 
    "Conspiracy claims that the moon landing was staged in a movie studio, but evidence proves otherwise."
}

# Sidebar: select sample news
st.sidebar.title("Or try a sample news article:")
sample_selected = st.sidebar.selectbox("Select a sample:", list(sample_news.keys()))
text = st.text_area("Paste the news text here:", value=sample_news[sample_selected])

# Function to get top contributing words
def get_top_words(text, model, vectorizer, n=10):
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_indices = np.argsort(np.abs(coefs))[::-1][:n]
    top_words = [feature_names[i] for i in top_indices if feature_names[i] in text.lower().split()]
    return top_words if top_words else ["None"]

# Function to create PDF
def create_pdf(result_text, confidence, article_text, top_words):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Fake News Detection Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, f"Prediction: {result_text}")
    pdf.ln(2)
    pdf.multi_cell(0, 8, f"Confidence: {confidence}%")
    pdf.ln(5)
    pdf.multi_cell(0, 8, "News Text:")
    pdf.ln(2)
    pdf.multi_cell(0, 8, article_text)
    pdf.ln(5)
    pdf.multi_cell(0, 8, "Top Contributing Words: " + ", ".join(top_words))
    
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# Load dataset for metrics (if available)
try:
    df = pd.read_csv("fake_news_dataset.csv")  # Columns: 'text', 'label'
    X_test = vectorizer.transform(df['text'])
    y_test = df['label']
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
except FileNotFoundError:
    st.warning("Dataset 'fake_news_dataset.csv' not found. Metrics will use dummy data.")
    acc, f1, cm = 0.50, 0.50, np.array([[0,0],[0,0]])

# Predict button
if st.button("Predict"):
    if text.strip() != "":
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = round(max(proba) * 100, 2)
        result_text = "REAL NEWS" if prediction == 1 else "FAKE NEWS"
        
        # Display result
        if prediction == 1:
            st.success(f"✅ Prediction: {result_text} (Confidence: {confidence}%)")
        else:
            st.error(f"❌ Prediction: {result_text} (Confidence: {confidence}%)")
        
        st.info("Tip: Check the source, author, and date for verification.")
        
        # Top contributing words
        top_words = get_top_words(text, model, vectorizer)
        st.write("📝 Top Contributing Words:", ", ".join(top_words))
        
        # Highlight contributing words in article
        highlighted_text = text
        for w in top_words:
            highlighted_text = highlighted_text.replace(w, f"**{w}**")
        st.write("📌 Highlighted Article:")
        st.markdown(highlighted_text)
        
        # PDF download
        pdf_file = create_pdf(result_text, confidence, text, top_words)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name="news_report.pdf",
            mime="application/pdf"
        )
        
        # Display metrics
        st.subheader("📊 Model Performance Metrics")
        st.write(f"Accuracy: {acc:.2f}, F1-score: {f1:.2f}")
        st.write("Confusion Matrix:")
        st.write(cm)
        
    else:
        st.warning("Please enter some text to classify.")
