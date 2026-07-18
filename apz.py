from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, re, string, pickle, os
import google.generativeai as genai
from functools import wraps
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'roman-urdu-sentiment-secret-key-2024'

DB = 'database.db'

#Gemini API Setup
genai.configure(api_key="AIzaSyAbMxydUNNUbYhf-Tyi3IOzbvdMKBKWjLo")   
gemini = genai.GenerativeModel("gemini-3-flash-preview")

#Load YOUR saved model & vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)


#Label mapping (from your notebook: positive=1, negative=0, neutral=2)
LABEL_MAP = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}

#YOUR EXACT stopwords from notebook
Roman_Urdu_stopwords = set([
    'hai', 'hain', 'tha', 'thi', 'the', 'aur', 'ya', 'ko',
    'ka', 'ki', 'ke', 'ne', 'se', 'mein', 'par', 'ek',
    'yeh', 'woh', 'main', 'ap', 'hum', 'tum', 'is', 'us',
    'kuch', 'bhi', 'hi', 'jo', 'jab', 'kab', 'kahan', 'kyun',
    'kya', 'nahi', 'na', 'ab', 'phir', 'lekin', 'magar'
])

#YOUR EXACT preprocess function from notebook
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in Roman_Urdu_stopwords]
    return ' '.join(words)

#Gemini sentiment function
def gemini_sentiment(text):
    try:
        prompt = f"""You are a Roman Urdu sentiment analysis expert.
Analyze the sentiment of the following Roman Urdu text.

Text: "{text}"

Reply in this EXACT format only, nothing else:
Sentiment: [Positive/Negative/Neutral]
Confidence: [High/Medium/Low]
Reason: [one sentence explanation in English]"""

        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {str(e)}"


# Database
def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL,
            email    TEXT    NOT NULL UNIQUE,
            password TEXT    NOT NULL,
            created  TEXT    NOT NULL
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            text         TEXT    NOT NULL,
            ml_sentiment TEXT    NOT NULL,
            ml_confidence REAL   NOT NULL,
            gemini_result TEXT   NOT NULL,
            created      TEXT    NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')
        db.commit()

init_db()

#Login required decorator
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

#Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm', '')

        if not all([name, email, password, confirm]):
            flash('All fields are required.', 'error')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('register.html')

        hashed = generate_password_hash(password)
        try:
            with get_db() as db:
                db.execute(
                    'INSERT INTO users (name,email,password,created) VALUES (?,?,?,?)',
                    (name, email, hashed, datetime.now().strftime('%Y-%m-%d %H:%M'))
                )
                db.commit()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('This email is already registered.', 'error')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        with get_db() as db:
            user = db.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id']   = user['id']
            session['user_name'] = user['name']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    with get_db() as db:
        history = db.execute(
            'SELECT * FROM history WHERE user_id=? ORDER BY id DESC LIMIT 10',
            (session['user_id'],)
        ).fetchall()
        total = db.execute(
            'SELECT COUNT(*) as c FROM history WHERE user_id=?',
            (session['user_id'],)
        ).fetchone()['c']

    return render_template('dashboard.html',
                           history=history,
                           total=total,
                           name=session['user_name'])

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.get_json()
    raw  = data.get('text', '').strip()

    if not raw:
        return jsonify({'error': 'Please enter some text.'}), 400

# Step 1: YOUR exact preprocessing from notebook
    cleaned = preprocess_text(raw)

#Step 2: YOUR exact model prediction from notebook
    text_vectorized  = tfidf.transform([cleaned])
    prediction_label = model.predict(text_vectorized)[0]        # returns 0, 1, or 2
    prediction_proba = model.predict_proba(text_vectorized).max()
    ml_sentiment     = LABEL_MAP[prediction_label]              # Negative / Positive / Neutral
    ml_confidence    = round(prediction_proba * 100, 1)

#Step 3: Gemini LLM analysis
    llm_result = gemini_sentiment(raw)

#Step 4: Save to database
    with get_db() as db:
        db.execute(
            '''INSERT INTO history
               (user_id, text, ml_sentiment, ml_confidence, gemini_result, created)
               VALUES (?,?,?,?,?,?)''',
            (session['user_id'], raw[:200], ml_sentiment,
             ml_confidence, llm_result,
             datetime.now().strftime('%Y-%m-%d %H:%M'))
        )
        db.commit()

    return jsonify({
        'ml_sentiment':  ml_sentiment,
        'ml_confidence': ml_confidence,
        'ml_label_num':  int(prediction_label),   # raw number: 0,1,2
        'cleaned':       cleaned,
        'llm_result':    llm_result
    })

if __name__ == '__main__':
    app.run(debug=True)
