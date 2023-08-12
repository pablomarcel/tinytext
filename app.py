from flask import Flask, render_template, request, redirect
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load the text
try:
    with open("static/divine_comedy_short.txt", "r", encoding="utf8") as file:
        documents = file.readlines()
except Exception as e:
    app.logger.error("Error loading divine_comedy_short.txt: %s", e)


# Custom tokenizer for the TfidfVectorizer
def tokenize(text):
    tokens = [
        word
        for word in nltk.word_tokenize(text)
        if len(word) > 1 and not word.isnumeric()
    ]
    return tokens


# Load the pre-trained TfidfVectorizer model and tfidf_matrix from tfidf_model.pkl
vectorizer = TfidfVectorizer(stop_words="english", tokenizer=tokenize)
vectorizer.fit(documents)
try:
    with open("static/tfidf_matrix.pkl", "rb") as file:
        tfidf_matrix = pickle.load(file)
except Exception as e:
    app.logger.error(f"Error loading tfidf_matrix.pkl: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        if query:
            try:
                tfidf_query = vectorizer.transform([query])
                cosine_similarities = linear_kernel(tfidf_query, tfidf_matrix).flatten()
                related_docs_indices = cosine_similarities.argsort()[:-5:-1]

                results = [documents[i] for i in related_docs_indices]
                return render_template(
                    "results.html",
                    query=query,
                    results=zip(range(1, len(results) + 1), results),
                )
            except Exception as e:
                app.logger.error(f"Error during search processing: {e}")
                return "An error occurred while processing the search.", 500
        else:
            return redirect("/")
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=False)
