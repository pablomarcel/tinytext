from flask import Flask, render_template, request, url_for, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
import logging


# Ensure nltk resources are downloaded
# nltk.download('punkt')

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

try:
    # Load the text
    with open('static/divine_comedy_short.txt', 'r', encoding="utf8") as file:
        documents = file.readlines()
except Exception as e:
    app.logger.error("Error loading divine_comedy.txt: %s", e)

# Custom tokenizer for the TfidfVectorizer
def tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text) if len(word) > 1 and not word.isnumeric()]
    return tokens

try:
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
    tfidf_matrix = vectorizer.fit_transform(documents)
except Exception as e:
    app.logger.error("Error during vectorization: %s", e)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            query = request.form['query']
            if query:
                tfidf_query = vectorizer.transform([query])
                cosine_similarities = linear_kernel(tfidf_query, tfidf_matrix).flatten()
                related_docs_indices = cosine_similarities.argsort()[:-5:-1]

                results = [documents[i] for i in related_docs_indices]
                return render_template("results.html", query=query, results=zip(range(1, len(results) + 1), results))
            else:
                return redirect(url_for('index'))
    except Exception as e:
        app.logger.error("Error during search processing: %s", e)
        return "An error occurred while processing the search."

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
