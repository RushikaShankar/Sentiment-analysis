import nltk
from nltk.tokenize import word_tokenize
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# ----------------------------
# Step 1: Download IMDb dataset
# ----------------------------
print("Downloading IMDb dataset...")
from sklearn.datasets import load_files
import nltk

if not os.path.exists("imdb"):
    nltk.download('movie_reviews')
    from nltk.corpus import movie_reviews
    os.mkdir("imdb")
    os.mkdir("imdb/pos")
    os.mkdir("imdb/neg")

    # Save files locally
    for fileid in movie_reviews.fileids():
        category = movie_reviews.categories(fileid)[0]
        review_text = movie_reviews.raw(fileid)
        filename = fileid.split("/")[-1] + ".txt"
        with open(f"imdb/{category}/{filename}", "w", encoding="utf8") as f:
            f.write(review_text)

# --------------------------------------
# Step 2: Load and split the IMDB dataset
# --------------------------------------
print("Loading dataset...")
data = load_files("imdb", shuffle=True, encoding="utf-8")
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# -----------------------
# Step 3: Train the model
# -----------------------
print("Training TF-IDF + Logistic Regression model...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# --------------------------
# Step 4: Evaluate the model
# --------------------------
X_test_vec = vectorizer.transform(X_test)
pred = model.predict(X_test_vec)

print("\nModel Evaluation:\n")
print(classification_report(y_test, pred, target_names=["Negative", "Positive"]))

# Save model for later use
#joblib.dump((vectorizer, model), "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "sentiment_model.pkl")


print("\nModel saved as sentiment_model.pkl")
