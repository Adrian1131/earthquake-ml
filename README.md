# 🌍 Earthquake Severity Prediction using Machine Learning (2001–2023)
A machine learning system to classify and analyze earthquake severity based on real-world seismic data using Python, AI, and data science techniques.

## 📌 Project Overview
This project applies supervised machine learning algorithms to seismic data from 2001 to 2023 to classify earthquake severity into four alert levels (green, yellow, orange, red). The ultimate goal is to build a data-driven tool that can assist early warning systems, emergency response, and public safety efforts in high-risk regions.

## We implemented multiple classification models including:

🌲 Random Forest

🤖 Artificial Neural Network (ANN)

📈 Support Vector Machine (SVM)

We explored the challenges of class imbalance, missing data, and realistic modeling of rare catastrophic events.

## 👥 Team Members

Adrian Flores – Data preprocessing, model development, poster creation, result analysis

Fabrice Polica – Feature engineering, model research, final report writing

## 🧠 Key Objectives
* Predict earthquake severity using AI techniques

* Use Python's data science libraries for preprocessing, visualization, and modeling

* Handle real-world constraints like missing values and class imbalance

* Support real-world disaster risk analysis through interpretable models

## 📊 Dataset
* Source: Kaggle - Earthquake Dataset (2001–2023)

* ~782 seismic events from over 780 locations

* Features include magnitude, depth, location, alert level, and instrumental intensities

### Features Used
* magnitude, depth, latitude, longitude, cdi, mmi, sig, tsunami, alert (target)

* Geolocation and metadata (e.g., country, continent, location)

* Preprocessing included encoding, imputation, and clustering (K-Means) for spatial insight

## 🧪 Methodology
### Preprocessing
* Missing value handling (drop/impute)

* Feature encoding (label encoding and one-hot encoding)

* K-Means clustering for geographic analysis

* Class balancing using manual oversampling and weighting

## Classifiers
* Random Forest

* Artificial Neural Network (Scikit-learn MLPClassifier)

* Support Vector Machine

* K-Means Clustering (for unsupervised spatial analysis)

* Evaluation Metrics
Accuracy

* Precision

* Recall

* F1 Score

## 📈 Results Summary
Test	Methodology	Key Takeaways
1.	Drop missing data + base model tuning	Good accuracy (86%), poor minority class recall
2.	Value imputation + class weighting	Better recall, but underrepresentation remained
3.	Full imputation + manual oversampling	Best balance across classes, improved F1 scores

**Final model performance improved for rare/severe earthquakes (Class 3), but limitations remained due to data size and imbalance.**

## 🔍 Analysis & Insights
* Real-world seismic data is naturally imbalanced — low-risk quakes dominate.

* Accuracy alone is misleading; recall and F1-score matter for rare class detection.

* Manual balancing helped reveal model learning capabilities under constrained data.

## 🚧 Limitations
* Small dataset (782 entries)

* Severe class imbalance

* High variance in test set performance depending on approach

* Risk of overfitting with oversampling

## 🔮 Future Work
* Expand dataset with new seismic events

* Incorporate deep learning (e.g., LSTM for time-based data)

* Add topological and geological features

* Use real-time data pipelines for live prediction

* Explore advanced ensemble and boosting models

## 🛠️ Tech Stack
### Languages: Python

### Libraries:

* pandas, numpy — Data manipulation

* matplotlib, seaborn — Visualization

* scikit-learn — ML models and preprocessing

* KMeans, SVM, RandomForestClassifier, MLPClassifier
