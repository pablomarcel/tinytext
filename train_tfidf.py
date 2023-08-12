from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pickle

# Custom tokenizer for the TfidfVectorizer
def tokenize(text):
    tokens = [
        word
        for word in nltk.word_tokenize(text)
        if len(word) > 1 and not word.isnumeric()
    ]
    return tokens

# Load the text
with open("static/divine_comedy_short.txt", "r", encoding="utf8") as file:
    documents = file.readlines()

vectorizer = TfidfVectorizer(stop_words="english", tokenizer=tokenize)
tfidf_matrix = vectorizer.fit_transform(documents)

# Save just the tfidf_matrix to a file
with open("static/tfidf_matrix.pkl", "wb") as matrix_file:
    pickle.dump(tfidf_matrix, matrix_file)
