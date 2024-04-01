# DataDiagnostics - Diabetes Prediction Machine Learning Models

## Project Overview

Diabetes is a prevalent chronic disease that affects millions of people worldwide. Early detection and prediction of diabetes are crucial for effective management and prevention of complications. This project aims to develop a predictive model to identify individuals at risk of developing diabetes based on various health indicators.

By leveraging machine learning algorithms and techniques, this model will analyse anonymous health records encompassing parameters such as BMI, age, blood pressure, dietary habits, etc. The project will employ feature selection methods and experiment with various algorithms, including logistic regression, decision trees, random forests, and neural networks, to build a robust and accurate predictive tool.

The ultimate goal is to gain insights into the significant factors contributing to the prediction of diabetes occurrence. Through early detection and intervention, the project aims to contribute to better management and prevention of diabetes, thus improving public health outcomes. The main component of this project is a Jupyter Notebook (`diabetes_prediction.ipynb`), which is intended to be run in Google Colaboratory (Google Colab). The notebook demonstrates the process of predicting diabetes using various machine learning models, covering  data preprocessing, exploratory data analysis, model training, evaluation, and comparison.


## Contents
- [Requirements](#Requirements)
- [Getting Started](#Getting-Started)
- [Documentation](#Documentation)
- [Notebook Contents](#Notebook-Contents)
- [Ethical Considerations](#Ethical-Considerations)
- [Conclusion](#Conclusion)
- [References](#References)
- [Acknowledgments](#Acknowledgments)
- [Collaborators](#Collaborators)



## <a id="Requirements"></a>Requirements
- A web browser capable of running Google Colab
- A Google Account (for storage of data)

## <a id="Getting-Started"></a>Getting Started

**GitHub Repository**
1. Clone the repository:

   ```git clone https://github.com/bernardtse/diabetes_prediction.git```

2. Navigate to the project directory:

   ```cd diabetes_prediction```

**Running the notebook in Google Colab**

1. Open Google Colab by going to [https://colab.research.google.com/](https://colab.research.google.com/).

   - Upload the notebook file [`diabetes_prediction.ipynb`] to Google Colab.
   
      or
   
   - Open the notebook directly at [https://colab.research.google.com/github/bernardtse/diabetes_prediction/blob/main/DiabetesPrediction.ipynb](https://colab.research.google.com/github/bernardtse/diabetes_prediction/blob/main/DiabetesPrediction.ipynb).

2. Ensure that you have the necessary libraries installed. The notebook requires the following libraries:

   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn` / 'sklearn'
   - `tensorflow`
   - `keras-tuner`

   You can install these libraries in Google Colab by running the following code cell at the beginning of the notebook:

   ```!pip install pandas numpy matplotlib seaborn scikit-learn sklearn tensorflow keras-tuner```

3. Run the notebook cells sequentially to execute the code and interact with the project.

**Note**
The notebook is designed to run on the cloud environment provided by Google Colab. It utilises the `Kaggle API` for data fetching and `Spark` for data processing. Both are automatically set up within the Colab environment. Therefore, there is no need to install Spark and the Kaggle API locally.

## <a id="Documentation"></a>Documentation

- **README**: Project overview and the instructions of running the Jupyter notebook is available in [`README.md`](README.md).
- **Project Report**: The project design, data processing procedures, model building and evaluation, tuning and optimisation are included in [`report.md`](report.md). The results of each model is presented and compared.


## <a id="Notebook-Contents"></a>Notebook Contents

1. **Introduction:**
   - Overview of the problem and dataset.
   
2. **Data Preprocessing:**
   - Handling missing values
   - Converting categorical variables to numerical format
   - Feature scaling

3. **Exploratory Data Analysis:**
   - Visualising distributions of numerical variables
   - Correlation analysis
   
4. **Model Building:**
   - Logistic Regression
   - Random Forest
   - Decision Tree
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Neural Network
   
5. **Model Evaluation:**
   - Confusion matrix
   - Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC)
   - Precision-recall curve
   
6. **Hyperparameter Tuning:**
   - Hyperparameter Tuning by GridSearchCV
   
7. **Neural Network Optimisation:**
   - Hyperparameter optimisation by Keras Tuner

8. **Results and Summary Report:**
   - Evaluation summary of each model
   - Comparisons of different models


## <a id="Ethical-Considerations"></a>Ethical Considerations
In the development of a machine learning model for diabetes prediction using a Kaggle dataset, ethical considerations play a central role. It is essential to prioritise data privacy, maintain transparency in model development, and ensure equitable access to healthcare insights. Additionally, efforts to mitigate bias in data collection and algorithmic decision-making are vital for responsible and ethical deployment in healthcare contexts.

## <a id="Conclusion"></a>Conclusion

Through this project, we demonstrate the application of machine learning techniques for diabetes prediction. By evaluating multiple models and optimising their performance, we aim to provide valuable insights for healthcare professionals in diagnosing and managing diabetes effectively.

**Note:** This project serves as an educational resource and does not replace professional medical advice. Always consult a healthcare provider for diagnosis and treatment of medical conditions.

## <a id="References"></a>References
**Dataset**: https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019

## <a id="Acknowledgments"></a>Acknowledgments
We would like to thank all contributors and participants who have made this project possible.

## <a id="Collaborators"></a>Collaborators
- [Aysha Gheewala](https://github.com/AyshaGheewala)
- [Godswill Anyasor](https://github.com/AnyasorG)
- [Kehlani Khan](https://github.com/kehlanijaan)
- [Sum Yeung Bernard Tse](https://github.com/bernardtse)
