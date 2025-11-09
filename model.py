# ==========================
# Smart Health Monitor - Model Training
# ==========================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load Dataset
print("📥 Loading dataset...")
data = pd.read_csv('heart.csv')
print("✅ Dataset loaded successfully!\n")

# Explore Data
print("Dataset Shape:", data.shape)
print("Columns:", data.columns.tolist())
print("\nMissing Values:\n", data.isnull().sum())

# Quick statistics
print("\nData Summary:\n", data.describe())

# Visualizations (Optional)
sns.countplot(x='condition', data=data)
plt.title("Heart Disease Presence (1 = Yes, 0 = No)")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Split features and target
X = data.drop('condition', axis=1)
y = data['condition']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Model Training
# ==========================
print("\n🚀 Training Models...\n")

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_acc = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {lr_acc:.2f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_acc:.2f}")

# ==========================
# Evaluation
# ==========================
print("\n=== Logistic Regression Report ===")
print(classification_report(y_test, y_pred_lr))

print("\n=== Random Forest Report ===")
print(classification_report(y_test, y_pred_rf))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================
# Save Model and Scaler
# ==========================
joblib.dump(rf, 'heart_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n✅ Model and Scaler saved successfully as 'heart_model.pkl' and 'scaler.pkl'!")
