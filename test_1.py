import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
'''
Initial libraries for each of our tests, copying for easier use. 
'''

df = pd.read_csv('earthquake_data.csv')

def clean_earthquake_data(df):
    features_to_keep = ['magnitude', 'date_time', 'cdi', 'mmi', 'tsunami',
                        'sig', 'nst', 'dmin', 'gap', 'magType', 'depth',
                        'latitude', 'longitude', 'continent', 'country',
                        'alert']

    # Keep only the specified features
    df = df[features_to_keep]

    # Drop rows missing critical values
    df = df.dropna(subset=['magnitude', 'depth', 'sig', 'latitude', 'longitude', 'alert'])

    # Fill less critical missing values with median or default values
    df.fillna({
        'dmin': df['dmin'].median(),
        'gap': df['gap'].median(),
        'mmi': df['mmi'].median(),
        'cdi': df['cdi'].median(),
        'tsunami': 0,
        'nst': df['nst'].median()
    }, inplace=True)

    # Process date_time correctly
    df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['hour'] = df['date_time'].dt.hour

    # Drop the original date_time column
    df.drop(columns=['date_time'], inplace=True)

    # Encode categorical columns
    for col in ['magType', 'continent', 'country']:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))

    # Encode the target column
    target_encoder = LabelEncoder()
    df['alert'] = target_encoder.fit_transform(df['alert'])

    # Final X and y
    X = df.drop(columns=['alert'])
    y = df['alert']

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feature_cols = df.columns.drop('alert')
    df[feature_cols] = X_scaled

    return X, y, target_encoder

X, y, target_encoder = clean_earthquake_data(df)

print("------------DATA CLEANED-------------")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("First 5 rows of X:\n", X.head())
print("First 5 rows of y:\n", y.head())

# Seeing the distribution of the data
print("Label Distribution: ")
print(y.value_counts())

"""
Updating the training split so it can have stratify, due to under representation of the classes.
"""
from sklearn.model_selection import train_test_split

# Making the train/test split including stratification.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y # CRUCIAL
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Testing the models (First Run, with class weighting)
random_forest_model = RandomForestClassifier(class_weight='balanced', random_state=42)
random_forest_model.fit(X_train, y_train)
random_forest_predictions = random_forest_model.predict(X_test)

"""
Temporary RF test for feature selection, 
"""
rf_temp = RandomForestClassifier(random_state=42)
rf_temp.fit(X_train, y_train)
feature_importances = pd.Series(rf_temp.feature_importances_, index=X.columns)
print(feature_importances.sort_values(ascending=False))

print("\n=== Random Forest Evaluation ===")
print(classification_report(y_test, random_forest_model.predict(X_test)))

"""
Working with the Support Vector Machine (SVM) model.
"""

svm_model = SVC(class_weight='balanced', kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

print("\n=== Support Vector Machine Evaluation ===")
print(classification_report(y_test, svm_model.predict(X_test)))

"""
Training the Neural Network model.
"""

ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=10000, random_state=42)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
ann_model.fit(X_train, y_train)

print("\n=== Artificial Neural Network Evaluation ===")
print(classification_report(y_test, ann_model.predict(X_test)))