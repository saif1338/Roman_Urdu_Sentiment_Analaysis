# 🇵🇰 Roman Urdu Sentiment Analyzer — Full Auth Web App

A complete Flask web application with user registration, login, and Roman Urdu sentiment analysis.

---

## 📁 Project Structure

```
urdu_auth_app/
│
├── app.py                   ← Main Flask app (auth + predict routes)
├── requirements.txt         ← All dependencies
├── database.db              ← SQLite DB (auto-created on first run)
├── model.pkl                ← Your trained ML model (add this)
├── vectorizer.pkl           ← Your trained TF-IDF vectorizer (add this)
│
└── templates/
    ├── landing.html         ← Home/marketing page
    ├── register.html        ← Registration form
    ├── login.html           ← Login form
    └── dashboard.html       ← Main app (analyzer + history)
```

---

## 🔄 User Flow

```
Landing Page (/) 
    ↓
Register (/register) → Creates account in SQLite DB
    ↓
Login (/login) → Sets session cookie
    ↓
Dashboard (/dashboard) → Analyze text + view history
    ↓
Logout (/logout) → Clears session
```

---

## ⚙️ Setup Instructions

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Add your trained model
```python
import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
```

### Step 3 — Connect your model in app.py
Uncomment these lines in app.py:
```python
model      = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
```
And inside the /predict route:
```python
vec        = vectorizer.transform([cleaned])
label      = model.predict(vec)[0]
confidence = float(model.predict_proba(vec).max())
```

### Step 4 — Run the app
```bash
python app.py
```
Visit: http://127.0.0.1:5000

---

## 🗄️ Database Tables

### users
| Column   | Type    | Description         |
|----------|---------|---------------------|
| id       | INTEGER | Primary key         |
| name     | TEXT    | Full name           |
| email    | TEXT    | Unique email        |
| password | TEXT    | Hashed password     |
| created  | TEXT    | Registration date   |

### history
| Column     | Type    | Description              |
|------------|---------|--------------------------|
| id         | INTEGER | Primary key              |
| user_id    | INTEGER | Foreign key → users      |
| text       | TEXT    | Input text               |
| sentiment  | TEXT    | Positive/Negative/Neutral|
| confidence | REAL    | Model confidence %       |
| created    | TEXT    | Analysis timestamp       |

---

## 🔒 Security Features
- Passwords are hashed using Werkzeug's `generate_password_hash`
- Sessions protected with Flask secret key
- `@login_required` decorator protects dashboard and predict routes
- SQL injection prevented via parameterized queries

---

## 👥 Team Roles

| Member | Task |
|--------|------|
| Member 1 | Dataset + EDA |
| Member 2 | Preprocessing |
| Member 3 | Train model → save .pkl files |
| Member 4 | Run this Flask app for demo |

---

Built with Flask · SQLite · Werkzeug · Scikit-learn
