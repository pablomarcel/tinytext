from flask import Flask, render_template, request, url_for, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
import os
from dotenv import load_dotenv

load_dotenv()


# Ensure nltk resources are downloaded
# nltk.download('punkt')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')


# Load the text
with open('static/divine_comedy.txt', 'r', encoding="utf8") as file:
    documents = file.readlines()


# Custom tokenizer for the TfidfVectorizer
def tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text) if len(word) > 1 and not word.isnumeric()]
    return tokens


vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
tfidf_matrix = vectorizer.fit_transform(documents)


@app.route('/', methods=['GET', 'POST'])
def index():
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
    return render_template('index.html')


if __name__ == "__main__":
    app.run()
