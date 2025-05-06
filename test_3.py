import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample, compute_sample_weight
from sklearn.impute import SimpleImputer

# === Load and Clean Data ===
df = pd.read_csv('earthquake_data.csv')
data = df.copy()

# === Numeric Imputation ===
numeric_cols = ['depth', 'sig', 'dmin', 'gap', 'mmi', 'cdi', 'nst', 'magnitude']
num_imputer = SimpleImputer(strategy='median')
data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])

# === Categorical Imputation ===
categorical_cols = ['alert', 'magType', 'continent', 'country', 'location', 'net', 'title']
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# === Datetime Processing ===
data['date_time'] = pd.to_datetime(data['date_time'], errors='coerce', dayfirst=True)
data['year'] = data['date_time'].dt.year
data['month'] = data['date_time'].dt.month
data['day'] = data['date_time'].dt.day
data['hour'] = data['date_time'].dt.hour
data.drop(columns=['date_time'], inplace=True)

# === Encode Categorical ===
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# === Label Encoding for Target ===
le_alert = LabelEncoder()
data['alert'] = le_alert.fit_transform(data['alert'])

# === Feature Scaling ===
features = data.drop(columns=['alert'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
x = pd.DataFrame(features_scaled, columns=features.columns)
y = data['alert']

# === Split Original Dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

# === Manual Oversampling ===
train_data = X_train.copy()
train_data['label'] = y_train

# Separate majority and minorities
majority = train_data[train_data['label'] == 0]
minorities = [train_data[train_data['label'] == cls] for cls in train_data['label'].unique() if cls != 0]

# Upsample minorities
resampled = [majority]
for minority in minorities:
    upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42
    )
    resampled.append(upsampled)

train_balanced = pd.concat(resampled)
X_train_bal = train_balanced.drop(columns=['label'])
y_train_bal = train_balanced['label']

print("Balanced class distribution:\n", y_train_bal.value_counts())

# === Train Models ===

# Random Forest
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train_bal, y_train_bal)
rf_preds = rf_model.predict(X_test)
print("\n=== Random Forest ===")
print(classification_report(y_test, rf_preds))
print("Accuracy:", accuracy_score(y_test, rf_preds))

# SVM
svm_model = SVC(class_weight='balanced', probability=True, random_state=42)
svm_model.fit(X_train_bal, y_train_bal)
svm_preds = svm_model.predict(X_test)
print("\n=== SVM ===")
print(classification_report(y_test, svm_preds))
print("Accuracy:", accuracy_score(y_test, svm_preds))

# ANN
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_bal)
ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000000, random_state=42)
ann_model.fit(X_train_bal, y_train_bal)
ann_preds = ann_model.predict(X_test)
print("\n=== ANN ===")
print(classification_report(y_test, ann_preds))
print("Accuracy:", accuracy_score(y_test, ann_preds))
