import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('earthquake_data.csv')
print(df.isnull().sum())  # Check missing values

# 367 missing values in the alert section
# 576 missing values for the continent
# 298 missing values for the country section.


# Make a copy of the dataset
data = df.copy()

# Replacing values. 
numeric_cols = ['depth', 'sig', 'dmin', 'gap', 'mmi', 'cdi', 'nst', 'magnitude']
num_imputer = SimpleImputer(strategy='median')
data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])

# Categorical Imputation
categorical_cols = ['alert', 'magType', 'continent', 'country', 'location', 'net', 'title']
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

data['date_time'] = pd.to_datetime(data['date_time'], errors='coerce', dayfirst=True)
data['year'] = data['date_time'].dt.year
data['month'] = data['date_time'].dt.month
data['day'] = data['date_time'].dt.day
data['hour'] = data['date_time'].dt.hour
data = data.drop(columns=['date_time'])

# Label Encoding example
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

x = data.drop(columns=['alert'])
y = data['alert']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# === Random Forest ===
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Classification Report:\n", classification_report(y_test, rf_preds))

"""
The Models again, learned effectively, but the for the highest risk level, they failed 
for this model. Which again, means based on our method, it can be dangerous to follow,
since it is very good on the lower risk levels, and the models learning rate it well
done, there seemed to be a severe disconnect between the data and even after replacing
the null values. Hence, this is effective, but only data that is fully present, and 
has no constraints. The model mimiced similar issues as it did with the first test. 

This time, the RF failed to learn the data, and the SVM failed completely. This was
due to the fact that the data was still unbalanced, and the SVM was not able to
learn the data during this approach. 
"""



# Training the support vector machine 
svm_model = SVC(class_weight='balanced')
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)
"""
Tried to balance the data, and the values were still way too unbalanced for SVM.
"""
print("\n=== Support Vector Machine ===")
print("Classification Report:\n", classification_report(y_test, svm_y_pred))
print("SVM Accuracy: ", accuracy_score(y_test, svm_y_pred))
# Model failed completely. This is specific to the accuracy and recall, not the data imbalance.

print(y.value_counts()) # -> Still unbalanced.


"""
Training the ANN again.
"""

ann_model = MLPClassifier(max_iter= 10000, random_state=42)
ann_model.fit(X_train, y_train)
ann_y_pred = ann_model.predict(X_test)

print("\n=== Artificial Neural Network ===")
print("\nANN Classification Report:\n", classification_report(y_test, ann_y_pred))
print("ANN Accuracy: ", accuracy_score(y_test, ann_y_pred))
