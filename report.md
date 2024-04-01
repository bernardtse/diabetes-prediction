# DataDiagnostics - Diabetes Prediction Machine Learning Models Report

## Contents
- [Introduction](#Introduction)
- [Dataset](#Dataset)
- [Data Prepocessing](#Data-Preprocessing)
- [Model Building and Tuning](#Models)
- [Results](#Results)
- [Conclusion](#Conclusion)
- [References](#References)


## <a id="Introduction"></a>**Introduction**



## <a id="Dataset"></a>**Dataset**

The dataset utilised in this project was curated by Neha Prerna Tigga and Dr. Shruti Garg from the Department of Computer Science and Engineering at BIT Mesra, Ranchi-835215, for research purposes only and is not intended for commercial use. An article detailing the implementation of this dataset has been published, providing further information and citation guidelines:



## <a id="Data-Preprocessing"></a>**Data Prepocessing**

**Data Preprocessing:**
   - Handling missing values: Imputed missing values using the mean for numerical features and the mode for categorical features.
   - Converting categorical variables to numerical format: Applied one-hot encoding to categorical variables.
   - Feature scaling: Standardised numerical features using Z-score normalisation.
   
**Exploratory Data Analysis (EDA):**
   - Visualising distributions of numerical variables: Utilised histograms and box plots to explore the distribution of features such as glucose level, blood pressure, BMI, etc.
   - Correlation analysis: Calculated Pearson correlation coefficients between features to identify potential correlations with the target variable (diabetes).
   

## <a id="Models"></a>**Model Building and Tuning**

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


## <a id="Results"></a>**Results**
   - Presented the evaluation summary of each model, including their accuracy, precision, recall, F1-score, ROC-AUC, and precision-recall AUC.
   - Compared the performance of different models and discussed their strengths and weaknesses.


This report evaluates the performance of various machine learning models in predicting the presence or absence of diabetes using a provided dataset. The models considered include Neural Network, SVM (Support Vector Machine), Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN).

| Model | Accuracy | Precision (Diabetes Absent) | Precision (Diabetes Present) | Recall (Diabetes Absent) | Recall (Diabetes Present) | F1-score (Diabetes Absent) | F1-score (Diabetes Present) |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| Neural Network | 97.24% | 97% | 98% | 99% | 92% | 98% | 95% |
| SVM | 87.29% | 90% | 80% | 93% | 71% | 91% | 75% |
| Logistic Regression | 85.64% | 88% | 78% | 93% | 65% | 90% | 71% |
| Decision Tree | 96.13% | 96% | 96% | 98% | 90% | 97% | 93% |
| Random Forest | 96.13% | 97% Precision | 94% | 98% | 92% | 97% | 93% |
| KNN | 96.13% | 97% | 94% | 98% | 92% | 97% | 93% |



## <a id="Conclusion"></a>**Conclusion**

The Neural Network model demonstrates the highest accuracy and balanced performance in predicting both classes. Decision Tree, Random Forest, and KNN also show promising results, closely following the Neural Network in terms of accuracy and performance metrics. SVM, and Logistic Regression exhibit lower accuracy and performance metrics compared to the other models. Based on the results, the Neural Network model is recommended for accurate and reliable diabetes prediction.

Through this project, we demonstrate the application of machine learning techniques for diabetes prediction. By evaluating multiple models and optimizing their performance, we aim to provide valuable insights for healthcare professionals in diagnosing and managing diabetes effectively.

**Note:** This project serves as an educational resource and does not replace professional medical advice. Always consult a healthcare provider for diagnosis and treatment of medical conditions.


## <a id="References"></a>**References**
- **Dataset:** https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019
- **Research Paper For the dataset:** Tigga, N. P., & Garg, S. (2020). Prediction of Type 2 Diabetes using Machine Learning Classification Methods. Procedia Computer Science, 167, 706-716. DOI: https://doi.org/10.1016/j.procs.2020.03.336
- **Diabetes Facts & figures - International Diabetes Federation:** https://idf.org/about-diabetes/diabetes-facts-figures/