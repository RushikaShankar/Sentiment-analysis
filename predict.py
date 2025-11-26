import joblib

# Load model + vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")


def predict_sentiment(text):
    """Return sentiment label for input text."""
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    return "Positive" if prediction == 1 else "Negative"


# Example usage:
if __name__ == "__main__":
    sample = "I love this product!"
    print("Text:", sample)
    print("Predicted Sentiment:", predict_sentiment(sample))
