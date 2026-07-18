import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

np.random.seed(42)

# Heart Disease Dataset
n_samples = 500
df = pd.DataFrame({
    'Age': np.random.randint(25, 80, n_samples),
    'Cholesterol': np.random.randint(150, 400, n_samples),
    'Blood_Pressure': np.random.randint(90, 180, n_samples),
    'Blood_Sugar': np.random.randint(70, 200, n_samples),
    'Heart_Rate': np.random.randint(60, 120, n_samples),
    'Exercise': np.random.randint(0, 300, n_samples),
    'Smoking': np.random.randint(0, 2, n_samples),
})

# Create disease target (Heart Disease)
df['Disease'] = (
    (df['Cholesterol'] > 240) | 
    (df['Blood_Pressure'] > 140) | 
    (df['Age'] > 60) & (df['Heart_Rate'] > 100)
).astype(int)

print("=" * 50)
print("DISEASE PREDICTION MODEL")
print("=" * 50)
print(f"\nDataset: {len(df)} samples")
print(f"Diseased: {df['Disease'].sum()} | Healthy: {(1-df['Disease']).sum()}")

# Split Data
X = df.drop('Disease', axis=1)
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='rbf', probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5)
}

print("\n" + "=" * 50)
print("MODEL RESULTS")
print("=" * 50)

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Model': name,
        'Accuracy': f'{accuracy:.4f}',
        'Precision': f'{precision:.4f}',
        'Recall': f'{recall:.4f}',
        'F1-Score': f'{f1:.4f}',
        'ROC-AUC': f'{roc_auc:.4f}'
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

# Results Table
results_df = pd.DataFrame(results)
print("\n" + "=" * 50)
print(results_df.to_string(index=False))
print("=" * 50)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Metrics Comparison
x = np.arange(len(models))
width = 0.15
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, metric in enumerate(metrics):
    values = [float(results[j][metric]) for j in range(len(models))]
    axes[0].bar(x + i*width, values, width, label=metric, color=colors[i])

axes[0].set_ylabel('Score')
axes[0].set_title('Disease Prediction Model Performance')
axes[0].set_xticks(x + width * 2)
axes[0].set_xticklabels(models.keys(), rotation=15, ha='right')
axes[0].legend(fontsize=8)
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)

# Feature Importance (Random Forest)
rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train_scaled, y_train)
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)

axes[1].barh(importance['Feature'], importance['Importance'], color='steelblue')
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importance - Random Forest')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()

import os
save_path = 'disease_prediction_results.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved: {os.path.abspath(save_path)}")

plt.show()

print("\n✓ Task 4 Complete!")
