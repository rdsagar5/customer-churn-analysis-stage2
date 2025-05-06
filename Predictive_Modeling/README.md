Deliverables for Stage 3 of Predictive Modelling

 All of the deliverables for the third stage of the Customer Churn Analysis project's predictive modelling are included in this folder.  Our objective was to use historical telecom customer data to develop and assess an Artificial Neural Network (ANN) that could forecast customer attrition.

 Contents:
 The ANN model's architecture, including input features, hidden layers, activation functions, loss function, optimiser, dropout, and evaluation metrics, is explained in depth in ANN_Architecture_Document.docx.
 - trained_ann_model.h5: The trained ANN model saved using Keras. - ann_model_training.py: Python script to preprocess the dataset, apply SMOTE for class balancing, train the ANN model, assess performance, and save the model as.h5.  This file is prepared for churn prediction pipeline deployment or integration.

 Performance of the Model:
 - Accuracy: 82%
 - F1 Rating: 0.78
 76% recall
 - AUC-ROC: 0.84

How to Use: 1. Set up the necessary libraries:

Installing pandas numpy scikit-learn imbalanced-learn tensorflow with pip

2. Prepare the Dataset: Make sure the dataset is in the same directory and has the name customer_churn_data.csv.

3. Launch the Python training script, ann_model_training.py.

This will create the trained_ann_model.h5 file and train the model.

Notes: To address class imbalance, SMOTE was employed.
EarlyStopping was put into place to stop overfitting.
- tf.keras.models.load_model() can be used to load the.h5 model.

**Group 2: ACS WIL Project Team Name Role Contributors**
Project manager and business analyst **Sagar B. K.**
**Samundra Giri**, a predictive modelling data analyst
**Amrita Katuwal**, Clustering Data Analyst
**Yadav Suruchi**, a data engineer
