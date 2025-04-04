import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # Changed to classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)

# Load the data
df = pd.read_csv("freMTPL2freq.csv").head(50000)

# Display data info for validation
print("Dataset info:")
print(f"Shape: {df.shape}")
print(df.head())
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Convert ClaimNb to binary target (0 if no claims, 1 if any claims)
df['ClaimNb_Binary'] = (df['ClaimNb'] > 0).astype(int)
print("\nTarget distribution:")
print(df['ClaimNb_Binary'].value_counts(normalize=True))

# Identify categorical and numerical columns
categorical_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
numerical_cols = ['Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']

# Select features and target
X = df[categorical_cols + numerical_cols]
y = df['ClaimNb_Binary']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create preprocessor with both categorical and numerical transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create pipeline with preprocessor and classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and preprocessor (included in the pipeline)
model_path = os.path.join('models', 'insurance_model.pkl')
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}")

# Example prediction
sample = X.iloc[0:1]
prediction_proba = model.predict_proba(sample)
prediction = model.predict(sample)[0]
actual = y.iloc[0]

print("\nExample prediction:")
print(f"Input features: {sample.iloc[0].to_dict()}")
print(f"Predicted claim probability: {prediction_proba[0][1]:.4f}")
print(f"Predicted claim (binary): {prediction}")
print(f"Actual claim (binary): {actual}")