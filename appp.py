from flask import Flask, render_template, request, jsonify
import pickle
import re
import os

app = Flask(__name__)

model=pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer=pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

class DemoModel:
    """Placeholder until you load your real trained model."""
    POSITIVE_WORDS = ['acha', 'zabardast', 'best', 'mast', 'badhiya',
                      'pyara', 'kamaal', 'shandaar', 'khushi', 'pasand',
                      'umda', 'wah', 'behtreen', 'lajawaab', 'dilchasp']
    NEGATIVE_WORDS = ['bura', 'ganda', 'bekar', 'kharab', 'bakwas',
                      'bori', 'ghalt', 'nafrat', 'faltu', 'wahiyat',
                      'afsos', 'mushkil', 'pareshan', 'takleef', 'bad']

    def predict(self, text):
        text_lower = text.lower()
        pos = sum(1 for w in self.POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in self.NEGATIVE_WORDS if w in text_lower)
        if pos > neg:
            return 'positive', round(0.65 + min(pos * 0.05, 0.30), 2)
        elif neg > pos:
            return 'negative', round(0.65 + min(neg * 0.05, 0.30), 2)
        else:
            return 'neutral', 0.55

demo_model = DemoModel()


ROMAN_URDU_STOPWORDS = [
    'hai', 'hain', 'tha', 'thi', 'the', 'aur', 'ya', 'ko',
    'ka', 'ki', 'ke', 'ne', 'se', 'mein', 'par', 'ek',
    'yeh', 'woh', 'main', 'ap', 'hum', 'tum', 'is', 'us',
    'kuch', 'bhi', 'hi', 'jo', 'jab', 'kab', 'kahan', 'kyun',
    'kya', 'nahi', 'na','ab', 'phir', 'lekin', 'magar'
]

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)       # remove numbers/symbols
    tokens = text.split()
    tokens = [w for w in tokens if w not in ROMAN_URDU_STOPWORDS]
    return ' '.join(tokens)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_text = data.get('text', '').strip()

    if not raw_text:
        return jsonify({'error': 'Please enter some text.'}), 400

    cleaned = preprocess(raw_text)

    #Swap this block with your real model
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()
    sentiment, confidence = demo_model.predict(cleaned)
    

    return jsonify({
        'sentiment': sentiment,
        'confidence': round(confidence * 100, 1),
        'cleaned_text': cleaned
    })


if __name__ == '__main__':
    app.run(debug=True)
