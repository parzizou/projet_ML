# SMOTE Integration Guide

This guide demonstrates how to integrate SMOTE into the ML training pipeline.

## Installation

```bash
pip install imbalanced-learn>=0.11.0,<0.13.0 scikit-learn>=1.3.0,<1.5.0
```

## Usage in Jupyter Notebook / Training Script

### Example: Complete Training Pipeline with SMOTE

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Import SMOTE handler
import sys
sys.path.insert(0, 'attrition-app')
from smote_handler import SMOTEHandler, SMOTEConfig, apply_smote

# Load your data
df = pd.read_csv('data/cleaned_data.csv')

# Separate features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split data BEFORE applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Original training set distribution:")
print(f"Class 0: {sum(y_train == 0)}")
print(f"Class 1: {sum(y_train == 1)}")

# Preprocessing pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit and transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ============================================================
# APPLY SMOTE - Only on training data!
# ============================================================

# Method 1: Using apply_smote function
X_train_balanced, y_train_balanced, stats = apply_smote(
    X_train_processed,
    y_train,
    sampling_strategy='auto',
    k_neighbors=5,
    random_state=42
)

# Method 2: Using SMOTEHandler class (more control)
# config = SMOTEConfig(
#     sampling_strategy='auto',
#     k_neighbors=5,
#     random_state=42
# )
# handler = SMOTEHandler(config)
# X_train_balanced, y_train_balanced = handler.fit_resample(X_train_processed, y_train)
# stats = handler.get_statistics()

print(f"\nBalanced training set distribution:")
print(f"Class 0: {sum(y_train_balanced == 0)}")
print(f"Class 1: {sum(y_train_balanced == 1)}")

# ============================================================
# Train the model on balanced data
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # Can still use this for extra balance
)

model.fit(X_train_balanced, y_train_balanced)

# ============================================================
# Evaluate on ORIGINAL (unbalanced) test set
# ============================================================

y_pred = model.predict(X_test_processed)
y_pred_proba = model.predict_proba(X_test_processed)

print("\n" + "=" * 60)
print("Classification Report (on original test set)")
print("=" * 60)
print(classification_report(y_test, y_pred))

print("\n" + "=" * 60)
print("Confusion Matrix")
print("=" * 60)
print(confusion_matrix(y_test, y_pred))

# ============================================================
# Save the model and metadata
# ============================================================

# Save model
joblib.dump(model, 'attrition-app/models/attrition_model.joblib')

# Save preprocessor
joblib.dump(preprocessor, 'attrition-app/models/attrition_preprocessor.joblib')

# Save metadata including SMOTE statistics
metadata = {
    'model_type': 'RandomForestClassifier',
    'training_date': pd.Timestamp.now().isoformat(),
    'features': {
        'numeric': numeric_features,
        'categorical': categorical_features
    },
    'smote_applied': True,
    'smote_statistics': stats,
    'class_distribution': {
        'original_train': {
            'class_0': int(sum(y_train == 0)),
            'class_1': int(sum(y_train == 1))
        },
        'balanced_train': {
            'class_0': int(sum(y_train_balanced == 0)),
            'class_1': int(sum(y_train_balanced == 1))
        }
    }
}

joblib.dump(metadata, 'attrition-app/models/attrition_metadata.joblib')

print("\n✅ Model, preprocessor, and metadata saved successfully!")
print(f"   SMOTE was applied: {metadata['smote_applied']}")
print(f"   Synthetic samples created: {stats['synthetic_samples']}")
```

## Important Notes

1. **Always apply SMOTE AFTER train/test split**: Never apply SMOTE before splitting to avoid data leakage
2. **Only apply to training data**: Never apply SMOTE to test or validation sets
3. **Apply AFTER preprocessing**: SMOTE works better with numerical features (after encoding categoricals)
4. **Evaluate on original test set**: Use the unbalanced test set to get realistic performance metrics

## API Usage

The web application provides API endpoints to configure SMOTE for future training:

```python
import requests

# Configure SMOTE
response = requests.post(
    'http://localhost:8000/api/smote/apply',
    json={
        'enabled': True,
        'sampling_strategy': 'auto',
        'k_neighbors': 5,
        'random_state': 42
    }
)
print(response.json())

# Get current configuration
response = requests.get('http://localhost:8000/api/smote/config')
print(response.json())
```

## Different Sampling Strategies

```python
# Auto - balance minority class to match majority
apply_smote(X, y, sampling_strategy='auto')

# Specific ratio - minority class will be 50% of majority
apply_smote(X, y, sampling_strategy=0.5)

# Custom dict - specify exact number for each class
apply_smote(X, y, sampling_strategy={1: 100})

# Minority - only balance the smallest class
apply_smote(X, y, sampling_strategy='minority')
```

## Troubleshooting

### Error: "Not enough neighbors"
If you get this error, reduce `k_neighbors`:
```python
apply_smote(X, y, k_neighbors=1)  # Minimum is 1
```

### Memory issues with large datasets
For very large datasets, consider using:
- BorderlineSMOTE
- SVMSMOTE
- ADASYN
from the `imblearn.over_sampling` module.
