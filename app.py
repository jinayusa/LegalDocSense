from flask import Flask, request, render_template
import pickle

import spacy

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function
def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def classify():
    if request.method == "POST":
        input_text = request.form["document"]
        cleaned_text = preprocess(input_text)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]
        return render_template("result.html", prediction=prediction)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
