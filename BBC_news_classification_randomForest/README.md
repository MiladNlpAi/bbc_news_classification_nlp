# BBC News Classification using Random Forest

This project focuses on classifying BBC news articles into different categories using Natural Language Processing (NLP) techniques and a Random Forest Classifier. The dataset includes thousands of news articles labeled into categories such as tech, business, sport, etc.

## 📁 Dataset

The dataset used is `bbc-text.csv`, which contains two columns:
- `category`: the label for each article (e.g., tech, sport, business)
- `text`: the full content of the article

## 🧪 Workflow Overview

1. **Exploratory Data Analysis**
   - Visualized the distribution of categories
   - Generated word clouds per category
   - Analyzed most common words

2. **Text Preprocessing**
   - Lowercasing
   - Removing punctuation
   - Tokenization
   - Stopword removal
   - Stemming and Lemmatization

3. **Feature Extraction**
   - TF-IDF vectorization

4. **Modeling**
   - Trained a `RandomForestClassifier`
   - Evaluated using Accuracy, Classification Report, Confusion Matrix

5. **Prediction Export**
   - Saved test set predictions into `predictions.csv`

## 📊 Results

- Model: **Random Forest Classifier**
- Vectorizer: **TF-IDF**
- Accuracy: Around ~90% on test data
- Visualizations:
  - Word Clouds per category
  - Barplot of most common words
  - Confusion Matrix of model performance

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/MiladNlpAi/bbc_news_classification_nlp.git
   cd bbc-news-classification-rf


