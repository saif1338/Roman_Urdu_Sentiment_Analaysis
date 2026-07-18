from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, re, os, pickle
from functools import wraps
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

DB = 'database.db'

# ─── Load Model (uncomment when you have trained model) ──────────────────────
model      = pickle.load(open('best_model.pkl', 'rb'))
#vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# ─── Demo Model ───────────────────────────────────────────────────────────────
class DemoModel:
    POS = ['acha','zabardast','best','mast','badhiya','pyara','kamaal',
           'shandaar','khushi','pasand','umda','wah','behtreen','lajawaab']
    NEG = ['bura','ganda','bekar','kharab','bakwas','bori','ghalt',
           'nafrat','faltu','wahiyat','afsos','mushkil','pareshan','bad']
    def predict(self, text):
        t = text.lower()
        p = sum(1 for w in self.POS if w in t)
        n = sum(1 for w in self.NEG if w in t)
        if p > n:   return 'Positive', round(0.65 + min(p*0.05, 0.30), 2)
        elif n > p: return 'Negative', round(0.65 + min(n*0.05, 0.30), 2)
        else:       return 'Neutral',  0.55

demo = DemoModel()
# ──────────────────────────────────────────────────────────────────────────────

STOPWORDS = ['hai','hain','tha','thi','the','aur','ya','ko','ka','ki','ke',
             'ne','se','mein','par','ek','yeh','woh','main','ap','hum','tum',
             'is','us','kuch','bhi','hi','jo','jab','kab','kahan','kyun',
             'kya','nahi','na','mat','ab','phir','lekin','magar']

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(w for w in text.split() if w not in STOPWORDS)

# ─── Database ─────────────────────────────────────────────────────────────────
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
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            text      TEXT    NOT NULL,
            sentiment TEXT    NOT NULL,
            confidence REAL   NOT NULL,
            created   TEXT    NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')
        db.commit()

init_db()

# ─── Auth decorator ───────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name     = request.form.get('name','').strip()
        email    = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        confirm  = request.form.get('confirm','')

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
                db.execute('INSERT INTO users (name,email,password,created) VALUES (?,?,?,?)',
                           (name, email, hashed, datetime.now().strftime('%Y-%m-%d %H:%M')))
                db.commit()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'error')
            return render_template('register.html')

    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email','').strip().lower()
        password = request.form.get('password','')

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
    raw  = data.get('text','').strip()
    if not raw:
        return jsonify({'error': 'Please enter some text.'}), 400

    cleaned = preprocess(raw)

    # ── Swap with real model ──────────────────────────────────────────────
    # vec        = vectorizer.transform([cleaned])
    # label      = model.predict(vec)[0]
    # confidence = float(model.predict_proba(vec).max())
    label, confidence = demo.predict(cleaned)
    # ─────────────────────────────────────────────────────────────────────

    with get_db() as db:
        db.execute(
            'INSERT INTO history (user_id,text,sentiment,confidence,created) VALUES (?,?,?,?,?)',
            (session['user_id'], raw[:200], label,
             round(confidence*100,1),
             datetime.now().strftime('%Y-%m-%d %H:%M'))
        )
        db.commit()

    return jsonify({
        'sentiment':   label,
        'confidence':  round(confidence*100, 1),
        'cleaned':     cleaned
    })

if __name__ == '__main__':
    app.run(debug=True)
