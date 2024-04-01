# DataDiagnostics - Diabetes Prediction Machine Learning Models Report

## Contents
1. [Introduction](#Introduction)
2. [Dataset](#Dataset)
3. [Data Prepocessing](#Data-Preprocessing)
4. [Model Building and Tuning](#Models)
5. [Results](#Results)
6. [Conclusion](#Conclusion)
7. [References](#References)


## <a id="Introduction"></a>**1. Introduction**

According to International Diabetes Federation, approximately 537 million adults (20-79 years) are living with diabetes. Around 90% diabetes cases are Type 2, influenced by factors like economy, age, environment, and lifestyles. Preventive actions and early diagnosis can help mitigate diabetes impact. This project aims to predict Type 2 diabetes risk using machine learning algorithms based on lifestyle and family background. These accurate algorithms are essential in healthcare for risk assessment.

## <a id="Dataset"></a>**2. Dataset**

The dataset utilised in this project was curated by Neha Prerna Tigga and Dr. Shruti Garg from the Department of Computer Science and Engineering at BIT Mesra, Ranchi-835215, for research purposes only and is not intended for commercial use. An [article](https://www.sciencedirect.com/science/article/pii/S1877050920308024") detailing the implementation of this dataset has been published on [ScienceDirect](https://www.sciencedirect.com), providing further information. In the original The performance of Random Forest Classifier is found to be most accurate for both datasets.



## <a id="Data-Preprocessing"></a>**3. Data Prepocessing**

**Data Preprocessing:**
   - Handling missing values: Imputed missing values using the mean for numerical features and the mode for categorical features.
   - Converting categorical variables to numerical format: Applied one-hot encoding to categorical variables.
   - Feature scaling: Standardised numerical features using Z-score normalisation.
   
**Exploratory Data Analysis (EDA):**
   - Visualising distributions of numerical variables: Utilised histograms and box plots to explore the distribution of features such as glucose level, blood pressure, BMI, etc.
   - Correlation analysis: Calculated Pearson correlation coefficients between features to identify potential correlations with the target variable (diabetes).
   

## <a id="Models"></a>**4. Model Building and Tuning**

**Model Building:**
   - **Logistic Regression:** Trained a logistic regression model as a baseline model.
   - **Random Forest:** Implemented a random forest classifier to capture non-linear relationships between features.
   - **Decision Tree:** Built a decision tree classifier to understand the decision-making process.
   - **Support Vector Machine (SVM):** Employed SVM with different kernels (linear, polynomial, and radial basis function) to find the best separating hyperplane.
   - **K-Nearest Neighbors (KNN):** Implemented KNN to classify data points based on the majority class of their nearest neighbors.
   - **Neural Network:** Constructed a feedforward neural network with multiple hidden layers using TensorFlow/Keras.
   
**Model Evaluation:**
   - Confusion matrix: Analyzed true positive, false positive, true negative, and false negative predictions.
   - Metrics: Calculated accuracy, precision, recall, F1-score to assess model performance.
   - ROC curve and AUC-ROC: Plotted the Receiver Operating Characteristic (ROC) curve and calculated the Area Under the Curve (AUC) to evaluate model discrimination.
   - Precision-recall curve: Visualized the trade-off between precision and recall for different threshold values.
   
**Hyperparameter Tuning:**
   - Utilized GridSearchCV to tune hyperparameters for SVM, Random Forest, and Decision Tree models.
   
**Neural Network Optimization:**
   - Used Keras Tuner to perform hyperparameter optimization for the neural network model, including the number of hidden layers, neurons per layer, activation functions, and learning rate.


## <a id="Results"></a>**5. Results**
   - Presented the evaluation summary of each model, including their accuracy, precision, recall, F1-score, ROC-AUC, and precision-recall AUC.
   - Compared the performance of different models and discussed their strengths and weaknesses.


This report evaluates the performance of various machine learning models in predicting the presence or absence of diabetes using a provided dataset. The models considered include Neural Network, Support Vector Machine (SVM), Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN).

| Model | Accuracy | Precision (Diabetes Absent) | Precision (Diabetes Present) | Recall (Diabetes Absent) | Recall (Diabetes Present) | F1-score (Diabetes Absent) | F1-score (Diabetes Present) |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| Logistic Regression | 85.64% | 88% | 78% | 93% | 65% | 90% | 71% |
| Random Forest | 96.13% | 97% | 94% | 98% | 92% | 97% | 93% |
| Decision Tree | 96.13% | 96% | 96% | 98% | 90% | 97% | 93% |
| SVM | 87.29% | 90% | 80% | 93% | 71% | 91% | 75% |
| KNN | 96.13% | 97% | 94% | 98% | 92% | 97% | 93% |
| Neural Network | 96.69% | 97% | 96% | 98% | 92% | 98% | 94% |

### **Hyperparameter Tuning with GridSearchCV**
- **Logistic Regression:**
  - Best Parameters: {'C': 10, 'penalty': 'l2', 'solver': 'newton-cg'}
  - Best Accuracy: 0.8632854406130267
- **Random Forest:**
   - Best Parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
   - Best Accuracy: 0.9489080459770115
- **Decision Tree:**
   - Best Parameters: {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2}
   - Best Accuracy: 0.9488984674329501
- **SVM:**
  - Best Parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'poly'}
  - Best Accuracy: 0.9350670498084291
- **KNN:**
  - Best Parameters: {'algorithm': 'ball_tree', 'n_neighbors': 3, 'weights': 'distance'}
  - Best Accuracy: 0.9530459770114943

## <a id="Conclusion"></a>**6. Conclusion**

The Neural Network model demonstrates the highest accuracy and balanced performance in predicting both classes. Random Forest, Decision Tree and KNN also show promising results, closely following the Neural Network in terms of accuracy and performance metrics. SVM and Logistic Regression exhibit lower accuracy and performance metrics compared to the other models.

Hyperparameter Tuning by GridSearchCV give significant improvement to SVM and minor improvement to Logistic Regresssion. However, due to diminishing returns, no visible improvements are shown in Decision Tree, Random Forest and KNN after tuning.

Based on the results, the Neural Network model is recommended for accurate and reliable diabetes prediction. However, if computing resources is taken into account, Random Forest, Decision Tree and KNN are all viable options. These three models work well in striking a balance between accuracy and efficiency.

Through this project, we demonstrate the application of machine learning techniques for diabetes prediction. By evaluating multiple models and optimizing their performance, we aim to provide valuable insights for healthcare professionals in diagnosing and managing diabetes effectively.

**Note:** This project serves as an educational resource and does not replace professional medical advice. Always consult a healthcare provider for diagnosis and treatment of medical conditions.



## <a id="References"></a>**7. References**
- **Dataset:** https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019
- **Research Paper For the dataset:** Tigga, N. P., & Garg, S. (2020). Prediction of Type 2 Diabetes using Machine Learning Classification Methods. Procedia Computer Science, 167, 706-716. DOI: https://doi.org/10.1016/j.procs.2020.03.336
- **Diabetes Facts & figures - International Diabetes Federation:** https://idf.org/about-diabetes/diabetes-facts-figures/The Neural Network model demonstrates the highest accuracy and balanced performance in predicting both classes. Decision Tree, Random Forest, and KNN also show promising results, closely following the Neural Network in terms of accuracy and performance metrics. SVM, and Logistic Regression exhibit lower accuracy and performance metrics compared to the other models. Based on the results, the Neural Network model is recommended for accurate and reliable diabetes prediction.