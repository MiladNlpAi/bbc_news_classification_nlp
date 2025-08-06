import pandas as pd
import numpy as np
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re

# Load and inspect the dataset
df = pd.read_csv('/content/bbc-text.csv')
df.head()

# Quick category distribution check
sns.countplot(data=df, x='category', palette='viridis')
plt.title("Distribution of News Categories")
plt.xticks(rotation=45)
plt.show()

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools for text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Basic text cleaning + tokenization + stemming + lemmatization
def preprocess_text(text):
  text = text.lower()
  punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~،؛؟«»"""
  text = re.sub(f"[{re.escape(punctuation)}]", '', text)
  tokens = nltk.word_tokenize(text)
  tokens_with_out_stop = [word for word in tokens if word not in stop_words]
  stemmed_tokens = [stemmer.stem(word) for word in tokens_with_out_stop]
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
  return lemmatized_tokens

# Apply preprocessing to text column
df['preprocessed_text'] = df['text'].apply(preprocess_text)
df['preprocessed_text'] = df['preprocessed_text'].apply(lambda x: " ".join(x))
df[['text', 'preprocessed_text']].head()

from wordcloud import WordCloud

# Generate word clouds for each category
for category in df['category'].unique():
    category_text = " ".join(df[df['category'] == category]['preprocessed_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(category_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Word Cloud for {category}")
    plt.axis('off')
    plt.show()

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(category_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Word Cloud for {category}")
    plt.axis('off')
    plt.show()

# Analyze most common tokens after preprocessing
all_words = [word for text in df['preprocessed_text'] for word in text.split()]
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(20)
common_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

sns.barplot(data=common_df, x='Frequency', y='Word', palette='plasma')
plt.title("Most Common Words After Preprocessing")
plt.show()

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_text'])

# TF-IDF matrix shape
tfidf_matrix.shape

# Check top features in first doc
tfidf_sample = pd.DataFrame(tfidf_matrix[0].toarray().T, columns=["TF-IDF Score"])
tfidf_sample['Word'] = tfidf_vectorizer.get_feature_names_out()
tfidf_sample = tfidf_sample.sort_values(by="TF-IDF Score", ascending=False).head(10)

sns.barplot(data=tfidf_sample, x='TF-IDF Score', y='Word', palette='cividis')
plt.title("Top 10 TF-IDF Features for First Document")
plt.show()

# Encode text labels to numeric
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# Just to check encoding
df[['category', 'category_encoded']].drop_duplicates().sort_values('category_encoded')

# Features and labels
X = tfidf_matrix
y = df['category_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape

# Random Forest Classifier
classifier = RandomForestClassifier(min_samples_split=5, n_estimators=200, random_state=42)
classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = classifier.predict(X_test)

# Classification report to evaluate model performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix to visualize model performance
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}")

# Final preview of the data
df.head()

# Save predictions for analysis or inspection
test_indices = y_test.index
output_df = pd.DataFrame({
    'index': test_indices,
    'preprocessed_text': df.loc[test_indices, 'preprocessed_text'],
    'predicted_category': label_encoder.inverse_transform(y_pred)
})
output_df.to_csv('predictions.csv', index=False)
output_df.head()
