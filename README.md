# 7072CEM – Machine Learning Assignment  
### Human Activity Recognition (HAR) Using Smartphone Sensor Data  

This repository contains my full implementation for the **7072CEM – Machine Learning** coursework at Coventry University.  
The project focuses on building a multi-class classification model for **Human Activity Recognition (HAR)** using the **UCI HAR dataset**, applying machine learning algorithms, PCA-based feature reduction, and hyperparameter tuning using Grid Search.

---

## 📌 **Project Overview**

Human Activity Recognition (HAR) aims to automatically classify physical activities (such as walking or sitting) from smartphone sensor signals.  
In this project, multiple machine learning models were trained on the UCI HAR dataset to identify the following activities:

- **LAYING**  
- **SITTING**  
- **STANDING**  
- **WALKING**  
- **WALKING_DOWNSTAIRS**  
- **WALKING_UPSTAIRS**

The system uses **accelerometer and gyroscope signals** recorded from Samsung Galaxy S II smartphones with 561-feature vectors per observation.

---

## 📊 **Dataset Description**

**Dataset:** UCI Human Activity Recognition (HAR) Using Smartphones  
**Subjects:** 30 individuals  
**Signals:** 3-axis acceleration + 3-axis gyroscope  
**Windows:** 2.56s with 50% overlap  
**Features:** 561  
**Classes:** 6  
**Train/Test Split:** 70% training, 30% testing  

Dataset link (Official Repository):  
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The dataset includes preprocessed feature vectors derived from time-domain and frequency-domain signals.

---

## 🧠 **Machine Learning Models Used**

The following models were implemented and compared:

1. **Logistic Regression (Multinomial)**
2. **Support Vector Machine (SVM) – RBF Kernel**
3. **K-Nearest Neighbour (KNN)**
4. **Linear Discriminant Analysis (LDA)**
5. **Decision Tree Classifier**

A **Pipeline** was used for each model, consisting of:
- `StandardScaler()`
- `PCA()`
- Machine Learning Classifier

Hyperparameters were tuned using **GridSearchCV**, including PCA components.

---

## 🛠 **Technologies Used**
- Python 3  
- NumPy  
- Pandas  
- Scikit-Learn  
- Matplotlib & Seaborn  
- PCA for dimensionality reduction  
- GridSearchCV for model optimisation  

---

## ⚙️ **Training Procedure**

### **1. Data Preprocessing**
- Merged subject, features, and activity labels  
- Standardised features using **StandardScaler()**  
- Converted activity IDs into categorical labels  
- Applied PCA for dimensionality reduction (tested 30, 50, 70 components)

### **2. Model Training**
Each model was trained using:
- 5-Fold Cross-Validation  
- Accuracy scoring  
- PCA components included as tunable parameters

### **3. Evaluation Metrics**
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- PCA Visualisation (2D scatter plot)

---

## 🏆 **Results Summary**

| Model | Accuracy |
|-------|----------|
| **Logistic Regression** | **0.9557** |
| **SVM (RBF Kernel)** | **0.9531** |
| **KNN** | **0.9485** |
| **LDA** | **0.9207** |
| **Decision Tree** | **0.8350** |

### **🔍 Best Hyperparameters Found (Grid Search)**

- **Logistic Regression:**  
  `{ 'clf__C': 1, 'pca__n_components': 70 }`

- **SVM (RBF):**  
  `{ 'clf__C': 10, 'clf__gamma': 0.01, 'pca__n_components': 70 }`

- **KNN:**  
  `{ 'clf__n_neighbors': 3, 'clf__weights': 'distance', 'pca__n_components': 70 }`

- **LDA:**  
  `{ 'pca__n_components': 70 }`

- **Decision Tree:**  
  `{ 'clf__max_depth': 10, 'clf__min_samples_split': 2, 'pca__n_components': 30 }`

---

## 📉 **Visualisations Included**

This project includes all major visualisations required for a complete ML workflow:

### ✔ Confusion Matrices for each model  
- Logistic Regression  
- SVM RBF  
- KNN  
- LDA  
- Decision Tree  

### ✔ PCA (2D) Visualisation  
Shows clustering of activities after dimensionality reduction.

All figures are stored in the `/figures` directory.

---

## 📁 **Repository Structure**


