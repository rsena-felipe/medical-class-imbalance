# Handling Class Imbalance with medical data 



## Project Structure

The project is structured as follows: 

```bash
.
├── data
│   ├── breast_cancer
│   └── pima
├── models
│   ├── breast_cancer
│   └── pima
├── notebooks
│   ├── breast_cancer
│   │   ├── 1_EDA_FeatureEngineering.ipynb
│   │   ├── 2_TrainModels.ipynb
│   │   └── 3_EvaluateModels.ipynb
│   └── PIMA
│       ├── 1_EDA_FeatureEngineering.ipynb
│       ├── regularization_standard_Scaler
│       │   ├── 2_TrainModels.ipynb
│       │   └── 3_EvaluateModels.ipynb
│       ├── SMOTE
│       │   ├── 2_TrainModels.ipynb
│       │   └── 3_EvaluateModels.ipynb
│       ├── standard_scaler
│       │   ├── 2_TrainModels.ipynb
│       │   └── 3_EvaluateModels.ipynb
│       └── stratified_regularization_smote
│           ├── 2_TrainModels.ipynb
│           └── 3_EvaluateModels.ipynb
├── README.md
├── results
│   ├── breast_cancer
│       ├── EDA
│       ├── Evaluation
│   │   └── Training
│   └── pima
│       ├── EDA
│       ├── Evaluation
│       └── Training
└── src
    └── my_functions
        ├── build_features.py
        ├── evaluate.py
        ├── __init__.py
        ├── preprocess.py
        └── train.py
     
```        

## Content


- Data Folder: Preprocess Data (Standard Scalar) and Raw Data. 

- Models Folder: contains the final models trained with Random Search with the different ML models and sampling strategies. Models trained: Ada Boost, Gradient Boosting, Logistic Regression, Random Forest, SVC, and XGBoost. The sampling strategies used were ADASYN, Borderline, Random Oversampling, SMOTE, Standard (No sampling strategy, unbalanced data), Random Undersampling and SMOTE with Active Learning. 

- Notebooks Folder: Contains the training procedure to train the models (code). It is divided into 3 stages: EDA (Exploratory Data Analysis), Train Models (Hyperparameter Tuning). For PIMA dataset standard scaler and stratified regularization with SMOTE was tried.

- Results: Contains three folder EDA (Exploratory Data Analysis graphs), Training (training history of each model), Evaluation (metrics graphs to evaluate the models).

- src: UDF functions to call in the notebooks (to create graphs and clean data mostly)

