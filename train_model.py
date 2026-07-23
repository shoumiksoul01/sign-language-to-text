import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

CSV_PATH   = r"D:\sign_language_project\data\landmarks.csv"
MODEL_PATH = r"D:\sign_language_project\model\sign_model.pkl"

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Total samples: {len(df)}")
print(f"Labels found: {sorted(df['label'].unique())}")

X = df.drop('label', axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print("\nTraining Random Forest (200 trees)... please wait...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Training complete!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
print("\nPer-letter report:")
print(classification_report(y_test, y_pred))

os.makedirs(r"D:\sign_language_project\model", exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model, 'classes': model.classes_}, f)

print(f"\nModel saved to {MODEL_PATH}")