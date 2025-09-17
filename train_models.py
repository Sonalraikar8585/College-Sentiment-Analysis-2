import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("balanced_college_reviews.csv")

# Basic preprocessing
df['review'] = df['review'].astype(str)
# df['label'] = df['label'].map({'positive': 1, 'negative': 0})
df['label'] = df['text_sentiment'].map({'positive': 1, 'negative': 0})
df.dropna(subset=['label'], inplace=True)

# Vectorize text
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['review'])
y = df['label']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
