**Student Exam Result Prediction using Random Forest**__

This project uses a Random Forest Classifier to predict whether a student will pass or fail an exam based on three features: hours of study, attendance percentage, and the number of assignments submitted. The dataset was created manually to demonstrate a complete machine learning workflow on a small, structured dataset.

Overview

The project follows these steps:

Load the dataset using pandas

Review the data (preview, summary, and basic statistics)

Encode the target column (“Pass”) into numeric values

Prepare the input features and target values

Train a Random Forest Classifier from scikit-learn

Make predictions on the training data and on a new example

Check feature importance to see which factors influence the result the most

Dataset

The dataset contains records for different students with the following columns:

Hours_Study

Attendance

Assignments_Submitted

Pass (Yes/No, later encoded as 1/0)

Model

The model used in this project is RandomForestClassifier. The algorithm builds multiple decision trees and combines their outputs to produce a final prediction, which helps improve stability and accuracy.

Results

The model predicts pass/fail outcomes based on the three input features. It also provides feature importance values, showing which factors contribute most to the prediction.

Conclusion

This project provides a simple and clear example of how to prepare data, train a Random Forest model, make predictions, and interpret the results using feature importance.
