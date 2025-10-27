# train_model.py
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Download stopwords
nltk.download('stopwords')

# Sample balanced dataset (FAKE + REAL)
data = [
    # FAKE NEWS (label=0)
    ("Aliens invade New York City", "Reports claim aliens landed in NYC, causing panic. Scientists confirm false.", 0),
    ("Chocolate cures all diseases", "A viral post claims eating chocolate daily cures all illnesses; false.", 0),
    ("Politician secretly controls the moon", "Fake report circulates that politician has secret moon base.", 0),
    ("Time travel device invented in garage", "Social media spreads news about a man claiming he built a time machine, fake.", 0),
    ("Cure for COVID-19 found in household spices", "Turmeric and garlic cure COVID-19? Health authorities deny it.", 0),
    ("Teleportation device sold online", "Viral post claims teleportation machine sold online; completely false.", 0),
    ("Chocolate increases lifespan by 150 years", "False article claims eating chocolate daily makes humans live 150 years.", 0),
    ("Moon landing was staged", "Conspiracy theory claiming moon landing was filmed in Hollywood, fake news.", 0),
    ("Dinosaurs cloned successfully", "Viral news claims scientists cloned dinosaurs, entirely false.", 0),
    ("Fish can survive on land for 1 year", "Fake post claims fish can live on land for 365 days.", 0),

    # REAL NEWS (label=1)
    ("NASA launches new satellite", "NASA successfully launched a new weather satellite to monitor storms.", 1),
    ("New vaccine approved for influenza", "Health authorities approved a new flu vaccine for public use.", 1),
    ("Stock markets rise amid economic growth", "Markets show strong growth as economy recovers.", 1),
    ("Local school wins science competition", "Students from Springfield High win regional science competition.", 1),
    ("New species of bird discovered", "Researchers discovered a new bird species in the Amazon rainforest.", 1),
    ("City council approves park renovation", "City council approved budget for park renovation and facilities.", 1),
    ("Electric car sales increase in 2025", "Report shows increase in electric vehicle sales this year.", 1),
    ("Tech company releases AI tool", "Company X launches AI-powered productivity tool for offices.", 1),
    ("Global temperatures rise in 2025", "Climate report confirms global average temperatures rose this year.", 1),
    ("Breakthrough in cancer research", "Scientists report promising results in new cancer treatment.", 1),
]

df = pd.DataFrame(data, columns=["title", "text", "label"])

# Combine title + text
df['text'] = df['title'] + ". " + df['text']

# Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)

# Split
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate
pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# Save
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved!")
