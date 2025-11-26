# Sentiment Analysis Using Machine Learning

This project is a simple NLP (Natural Language Processing) implementation that predicts whether a given text expresses a Positive or Negative sentiment. It uses the IMDB movie reviews dataset, TF-IDF vectorization, and a Logistic Regression classifier.

## Project Objective

To build a machine learning model that can understand text and classify it into sentiment categories.

The goal is to demonstrate:

+ Basic NLP preprocessing

+ Feature extraction using TF-IDF

+ Training a ML classifier

+ Saving & loading models

+ Running predictions on new text

This project is suitable for academic assignments and demonstrations.

## 📂 Project Structure
sentiment-project/

│── imdb/

|  ├── pos/        # Positive training examples

|  ├── neg/        # Negative training examples

│── sentiment.py        # Model training file

│── predict.py          # Sentiment prediction file

│── sentiment_model.pkl # Saved trained model

│── vectorizer.pkl      # Saved TF-IDF vectorizer

│── README.md           # Project documentation


## Technologies Used

Python 3

scikit-learn (Logistic Regression, TF-IDF Vectorizer)

joblib (Model saving)

NLTK / basic preprocessing

VS Code / Jupyter Notebook

## Dataset

IMDB Movie Reviews Dataset

This dataset is one of the most popular for sentiment analysis.
It contains 50,000 movie reviews labeled as:

Positive (pos/)

Negative (neg/)

📌 Full Official Name
IMDB Large Movie Review Dataset (ACL IMDB Dataset)

Created by Andrew Maas et al. (Stanford University)


Each file contains:

A short movie review

Label:

   pos/ → Positive

   neg/ → Negative

These samples are used to train the classifier.


### Expected output:

Model saved as sentiment_model.pkl
Vectorizer saved as vectorizer.pkl

Sample:

Text: I love this movie so much!
Predicted Sentiment: Positive

## How It Works

1. Load dataset

2. Clean text (lowercase, remove symbols, remove stopwords)

3. Convert text to numbers using TF-IDF Vectorizer

4. Train Logistic Regression

5. Save the model for later use

6. Predict using user input

### Model Used
✔ Logistic Regression

   A simple but effective algorithm for binary classification.

✔ TF-IDF Vectorizer

   Converts raw text into feature vectors by measuring word importance.

### Sample Predictions
Input Text	Output
“I love this product!”	Positive
“This movie was terrible.”	Negative
“The experience was good overall.”	Positive


## Conclusion

This project demonstrates a clear and effective application of NLP for text sentiment classification.
It is easy to understand, simple to execute, and perfect for academic grading or a GitHub portfolio.
