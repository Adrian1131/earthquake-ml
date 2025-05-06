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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
"""
Attempting feature engineering focusing on geographical features, using this method on top of KMeans to
attempt to the see the data in a more format
"""

df = pd.read_csv('earthquake_data.csv')
data = df

data['continent'] = data['continent'].fillna('Unknown')
data['country'] = data['country'].fillna('Unknown')

kmeans = KMeans(n_clusters = 5, random_state = 42)
data['region'] = kmeans.fit_predict(data[['latitude', 'longitude']])

def magnitude_category(mag):
    if mag < 4.0:
        return 'Minor'
    elif 4.0 <= mag < 6.0:
        return 'Moderate'
    elif 6.0 <= mag < 7.0:
        return 'Strong'
    else:
        return 'Severe'
    
data['magnitude_cat'] = data['magnitude'].apply(magnitude_category)

# Analyze cluster characteristics
cluster_summary = data.groupby('region').agg({
    'magnitude': ['mean', 'median'],
    'depth': ['mean', 'median'],
    'sig': ['mean', 'median'],
    'country': 'nunique'
}).reset_index()

print(cluster_summary)

# Scatter plot of clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['longitude'], data['latitude'], c=data['region'], cmap='viridis', s=10)
plt.colorbar(label='Cluster')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('K-Means Clustering of Earthquake Regions')
plt.show()

