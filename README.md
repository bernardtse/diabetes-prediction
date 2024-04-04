![DataDiagnostics](images/header.png)
# Project Title: Development of a Diabetes Prediction Model using Machine Learning

## Contents
1. [Project Overview](#1-project-overview)
2. [Requirements](#2-requirements)
3. [Getting Started](#3-getting-started)
4. [Documentation](#4-documentation)
5. [Notebook Contents](#5-notebook-contents)
6. [Model Visualisation using Web APP](#6-model-visualisation-using-web-app)
7. [Ethical Considerations](#7-ethical-considerations)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)
10. [Acknowledgments](#10-acknowledgments)
11. [Collaborators](#11-collaborators)


## 1. Project Overview

Diabetes is a prevalent chronic disease affecting millions worldwide. Early detection and prediction are crucial for effective management and complication prevention. This project aims to develop a predictive model to identify individuals at risk of diabetes using various health indicators.

By leveraging machine learning algorithms, anonymous health records, including parameters like BMI, age, blood pressure, and dietary habits, are analysed. Feature selection methods and various algorithms such as Logistic Regression, Random Forest, Decision Tree, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Neural Network are employed to build a robust predictive tool.

The project's main goal is to gain insights into significant factors contributing to diabetes prediction, contributing to better management and prevention, and improving public health outcomes. Additionally, a user-friendly web application has been developed to make the predictive model accessible. This application allows users to input their health parameters and receive predictions of their diabetes risk, along with explanations and insights into contributing factors.


## 2. Requirements

- A web browser capable of running Google Colaboratory (Google Colab)
- A Kaggle account (for data fetching)
- A Google account (for storage of data)


## 3. Getting Started

### GitHub Repository

i. Clone the repository:

   ```git clone https://github.com/bernardtse/diabetes_prediction.git```

ii. Navigate to the project directory:

   ```cd diabetes_prediction```
   
### Kaggle

i. Sign in to [Kaggle](https://kaggle.com) `https://kaggle.com/`.

ii. Go to Settings.

iii. Under the API section, create and download token `kaggle.json`.

### Running the notebook in Google Colab

i. Sign in to [Google Colab](https://colab.research.google.com/) `https://colab.research.google.com/`.

ii. Upload the notebook file `DiabetesPrediction.ipynb` to Google Colab.

iii. Upload `kaggle.json` to the path specified in the Jupyter Notebook in Google Drive (Default path: `My Drive`). If `kaggle.json` was already installed previously, modify the `kaggle_token_path` variable in the notebook to specify the path for Kaggle token.

iv. Ensure that you have the necessary libraries installed. The notebook requires the following libraries:

   - `Pandas`
   - `Numpy`
   - `Matplotlib`
   - `seaborn`
   - `scikit-learn`
   - `TensorFlow`
   - `Keras-tuner`

   You can install these libraries in Google Colab by running the following code cell at the beginning of the notebook:

   ```!pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras-tuner```

v. Run the notebook cells sequentially to execute the code and interact with the project.

### Rationale for using Google Colab, Kaggle API and PySpark

The notebook is designed to run on the cloud environment provided by Google Colab. It utilises the Kaggle API for data fetching and Spark for data processing. Both are automatically set up within the Colab environment. Therefore, there is no need to install `kaggle` and `pyspark` locally.

The dataset was fetched via the Kaggle API on the cloud, which was processed with PySpark before being converted to a Pandas DataFrame. This approach offers several advantages over downloading the CSV file locally and then uploading it to Google Drive for processing with Pandas:

- **Integration**: By fetching the dataset directly from Kaggle to Google Colab using the Kaggle API, data acquisition process is streamlined, without the need to manually download and upload files.

- **Collaboration and Sharing**: Google Colab allows for easy sharing and collaboration on Jupyter notebooks, making it convenient for teams to work together on data analysis projects.

- **Scalability**: PySpark is designed to handle processing data in of different sizes efficiently. It can distribute the workload across multiple nodes in a cluster, enabling parallel processing. This is especially advantageous when dealing with potentially larger datasets that may not fit into memory on a single machine.


## 4. Documentation

- **README**: Project overview and the instructions for running the Jupyter notebook are available in [`README.md`](README.md).
- **Project Report**: The project design, data processing procedures, model building and evaluation, tuning and optimisation are included in [`report.md`](report.md). The results of each model is presented and compared.
- **Jupyter Notebook**: Loading of the dataset, data cleaning and pre-processing, model building, model evaluation and model optimisation are all included in a single Jupyter Notebook [`DiabetesPrediction.ipynb`](DiabetesPrediction.ipynb).

- Graphs generated by the Jupyter notebook are stored in the [`images/`](images/) folder.
- Raw dataset is stored in [`resources/diabetes_dataset__2019.csv`](resources/diabetes_dataset__2019.csv).
- Processed dataset is stored in [`resources/processed_dataset.csv`](resources/processed_dataset.csv).
- Model architecture for Neural Network is stored in [`webapp/model.json`](webapp/model.json).
- Model weights for Neural Network is stored in [`webapp/model_weights.h5`](webapp/model_weights.h5).


## 5. Notebook Contents

i. **Introduction:**
   - Overview of the problem and dataset.
   
ii. **Data Loading and Pre-processing:**
   - Handling missing values
   - Converting categorical variables to numerical format
   - Feature scaling

iii. **Exploratory Data Analysis:**
   - Visualising distributions of numerical variables
   - Correlation analysis
   
iv. **Model Building:**
   - Logistic Regression
   - Random Forest
   - Decision Tree
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Neural Network
   
v. **Model Evaluations:**
   - Confusion matrix
   - Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC)
   - Precision-recall curve
   
vi. **Hyperparameter Tuning:**
   - Hyperparameter Tuning with GridSearchCV
   
vii. **Neural Network Optimisation:**
   - Hyperparameter optimisation with Keras Tuner

viii. **Results and Summary Report:**
   - Evaluation summary of each model
   - Comparisons of different models

## 6. **Model Visualisation Using Web App**

- For the visualisation of the diabetes prediction model, a web application was developed using Python Flask for the backend, HTML and CSS for the frontend, and JavaScript (`script.js`) for interactivity stored in the [`webapp/`](webapp/) folder.

- The web application provides an intuitive interface for users to input their health parameters and receive predictions of their diabetes risk. It utilises the trained machine learning model to generate predictions based on the input data.

- The backend script (`app.py`) handles the prediction logic and communicates with the frontend through HTTP requests. The HTML files define the structure of the web pages, while CSS stylesheets are used for styling and layout. JavaScript (`script.js`) enhances user interactivity and handles dynamic content updates.

- To run the web application locally:
  - i. Ensure `Python` and `Flask` are installed on your system.
  - ii. Ensure a compatable version of `TensorFlow` (version 2.15) and `Keras` (version 2.15) is installed: `pip install tensorflow==2.15`
     - **Note:** `Keras` is installed automatically when `TensorFlow` is installed.
  - iii. Navigate to the `webapp/` directory in the terminal: `cd webapp`
  - iv. Run the following command to start the Flask server: `python app.py`
  - v. Open a web browser and go to [http://localhost:5000](http://localhost:5000) to access the web application.


## 7. Ethical Considerations

In the development of a machine learning model for diabetes prediction using a Kaggle dataset, ethical considerations play a central role. It is essential to prioritise data privacy, maintain transparency in model development, and ensure equitable access to healthcare insights. Additionally, efforts to mitigate bias in data collection and algorithmic decision-making are vital for responsible and ethical deployment in healthcare contexts.


## 8. Conclusion

Through this project, we demonstrated the application of machine learning techniques for diabetes prediction. By evaluating multiple models and optimising their performance, we aim to provide valuable insights for healthcare professionals in diagnosing and managing diabetes effectively.

**Disclaimer:** This project serves as an educational resource and does not replace professional medical advice. Always consult a healthcare provider for diagnosis and treatment of medical conditions.


## 9. References

- **Dataset - Kaggle:** [https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019](https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019)
- **Research Paper for the Dataset:** Tigga, N. P., & Garg, S. (2020). Prediction of Type 2 Diabetes using Machine Learning Classification Methods. Procedia Computer Science, 167, 706-716. DOI: [https://doi.org/10.1016/j.procs.2020.03.336](https://doi.org/10.1016/j.procs.2020.03.336)
- **Diabetes Facts & Figures - International Diabetes Federation:** [https://idf.org/about-diabetes/diabetes-facts-figures/](https://idf.org/about-diabetes/diabetes-facts-figures/)
- **Diabetes - NHS:** [https://www.nhs.uk/conditions/diabetes/](https://www.nhs.uk/conditions/diabetes/)

## 10. Acknowledgments

We would like to thank all contributors and participants who have made this project possible.


## 11. Collaborators

- [Aysha Gheewala](https://github.com/AyshaGheewala)
- [Godswill Anyasor](https://github.com/AnyasorG)
- [Kehlani Khan](https://github.com/kehlanijaan)
- [Sum Yeung Bernard Tse](https://github.com/bernardtse)
