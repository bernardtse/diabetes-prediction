# DataDiagnostics - Diabetes Prediction Machine Learning Models Report

### Contents
- [Key Features](#Key-Features)
- [Technical Stack](#Technical-Stack)
- [Getting Started](#Getting-Started)
- [Usage](#Usage)
- [Documentation](#Documentation)
- [Ethical Considerations](#Ethical-Considerations)
- [References](#References)
- [Acknowledgments](#Acknowledgments)
- [Collaborators](#Collaborators)

1. **Introduction:**
   - Overview of the problem and dataset.
   
2. **Data Preprocessing:**
   - Handling missing values: Imputed missing values using the mean for numerical features and the mode for categorical features.
   - Converting categorical variables to numerical format: Applied one-hot encoding to categorical variables.
   - Feature scaling: Standardised numerical features using Z-score normalisation.
   
3. **Exploratory Data Analysis (EDA):**
   - Visualising distributions of numerical variables: Utilised histograms and box plots to explore the distribution of features such as glucose level, blood pressure, BMI, etc.
   - Correlation analysis: Calculated Pearson correlation coefficients between features to identify potential correlations with the target variable (diabetes).
   
4. **Model Building:**
   - **Logistic Regression:** Trained a logistic regression model as a baseline model.
   - **Random Forest:** Implemented a random forest classifier to capture non-linear relationships between features.
   - **Decision Tree:** Built a decision tree classifier to understand the decision-making process.
   - **Support Vector Machine (SVM):** Employed SVM with different kernels (linear, polynomial, and radial basis function) to find the best separating hyperplane.
   - **K-Nearest Neighbors (KNN):** Implemented KNN to classify data points based on the majority class of their nearest neighbors.
   - **Neural Network:** Constructed a feedforward neural network with multiple hidden layers using TensorFlow/Keras.
   
5. **Model Evaluation:**
   - Confusion matrix: Analysed true positive, false positive, true negative, and false negative predictions.
   - Metrics: Calculated accuracy, precision, recall, F1-score to assess model performance.
   - ROC curve and AUC-ROC: Plotted the Receiver Operating Characteristic (ROC) curve and calculated the Area Under the Curve (AUC) to evaluate model discrimination.
   - Precision-recall curve: Visualised the trade-off between precision and recall for different threshold values.
   
6. **Hyperparameter Tuning:**
   - Utilised GridSearchCV to tune hyperparameters for SVM, Random Forest, and Decision Tree models.
   
7. **Neural Network Optimisation:**
   - Used Keras Tuner to perform hyperparameter optimisation for the neural network model, including the number of hidden layers, neurons per layer, activation functions, and learning rate.

8. **Results and Summary Report:**
   - Presented the evaluation summary of each model, including their accuracy, precision, recall, F1-score, ROC-AUC, and precision-recall AUC.
   - Compared the performance of different models and discussed their strengths and weaknesses.






## Project Overview
This project aims to develop a predictive model to identify individuals at risk of developing diabetes based on various health indicators. By leveraging machine learning algorithms and techniques, this model will analyse anonymous health records encompassing parameters such as BMI, age, blood pressure, physical health and dietary habits. By employing feature selection methods and experimenting with various algorithms including logistic regression, decision trees, random forests and neural networks, this project seeks to build a robust and accurate predictive tool. 

The ultimate goal is to gain insights into the significant factors contributing to the prediction of diabetes occurrence. Through early detection and intervention, the project aims to contribute to better management and prevention of diabetes, thus improving public health outcomes.This repository contains a Jupyter Notebook file (`diabetes_prediction.ipynb`) that demonstrates the process of predicting diabetes using various machine learning models. The notebook covers data preprocessing, exploratory data analysis, model training, evaluation, and comparison.

## Overview

Diabetes is a prevalent chronic disease that affects millions of people worldwide. Early detection and prediction of diabetes are crucial for effective management and prevention of complications. In this project, we explore the use of machine learning algorithms to predict the presence or absence of diabetes based on various health-related factors.

## Notebook Contents

1. **Introduction:**
   - Overview of the problem and dataset.
   
2. **Data Preprocessing:**
   - Handling missing values: Imputed missing values using the mean for numerical features and the mode for categorical features.
   - Converting categorical variables to numerical format: Applied one-hot encoding to categorical variables.
   - Feature scaling: Standardized numerical features using Z-score normalization.
   
3. **Exploratory Data Analysis (EDA):**
   - Visualizing distributions of numerical variables: Utilized histograms and box plots to explore the distribution of features such as glucose level, blood pressure, BMI, etc.
   - Correlation analysis: Calculated Pearson correlation coefficients between features to identify potential correlations with the target variable (diabetes).
   
4. **Model Building:**
   - **Logistic Regression:** Trained a logistic regression model as a baseline model.
   - **Random Forest:** Implemented a random forest classifier to capture non-linear relationships between features.
   - **Decision Tree:** Built a decision tree classifier to understand the decision-making process.
   - **Support Vector Machine (SVM):** Employed SVM with different kernels (linear, polynomial, and radial basis function) to find the best separating hyperplane.
   - **K-Nearest Neighbors (KNN):** Implemented KNN to classify data points based on the majority class of their nearest neighbors.
   - **Neural Network:** Constructed a feedforward neural network with multiple hidden layers using TensorFlow/Keras.
   
5. **Model Evaluation:**
   - Confusion matrix: Analyzed true positive, false positive, true negative, and false negative predictions.
   - Metrics: Calculated accuracy, precision, recall, F1-score to assess model performance.
   - ROC curve and AUC-ROC: Plotted the Receiver Operating Characteristic (ROC) curve and calculated the Area Under the Curve (AUC) to evaluate model discrimination.
   - Precision-recall curve: Visualized the trade-off between precision and recall for different threshold values.
   
6. **Hyperparameter Tuning:**
   - Utilized GridSearchCV to tune hyperparameters for SVM, Random Forest, and Decision Tree models.
   
7. **Neural Network Optimization:**
   - Used Keras Tuner to perform hyperparameter optimization for the neural network model, including the number of hidden layers, neurons per layer, activation functions, and learning rate.

8. **Results and Summary Report:**
   - Presented the evaluation summary of each model, including their accuracy, precision, recall, F1-score, ROC-AUC, and precision-recall AUC.
   - Compared the performance of different models and discussed their strengths and weaknesses.

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, keras-tuner

## Usage

1. Clone the repository:

git clone https://github.com/your-username/diabetes-prediction.git


2. Navigate to the project directory:

cd diabetes-prediction


3. Install the required libraries:

pip install -r requirements.txt


4. Open and run the Jupyter Notebook:

jupyter notebook diabetes_prediction.ipynb


5. Follow the instructions in the notebook to execute the code cells and analyze the results.

## Conclusion

Through this project, we demonstrate the application of machine learning techniques for diabetes prediction. By evaluating multiple models and optimizing their performance, we aim to provide valuable insights for healthcare professionals in diagnosing and managing diabetes effectively.

**Note:** This project serves as an educational resource and does not replace professional medical advice. Always consult a healthcare provider for diagnosis and treatment of medical conditions.

### Collaborators
- [Aysha Gheewala](https://github.com/AyshaGheewala)
- [Godswill Anyasor](https://github.com/AnyasorG)
- [Kehlani Khan](https://github.com/kehlanijaan)
- [Sum Yeung Bernard Tse](https://github.com/bernardtse)
