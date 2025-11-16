
import pandas as pd                      
from sklearn.ensemble import RandomForestClassifier   
from sklearn.preprocessing import LabelEncoder       

# read the CSV file into a DataFrame
df = pd.read_csv("student_results.csv")

# first few rows of the data to confirm it loaded correctly
print("Dataset preview using pandas:")
print(df.head())

# information about the dataset
print("\nDataset summary:")
print(df.info())

# Statistics for numeric columns
print("\nBasic statistics:")
print(df.describe())

# Convert the 'Pass' column (Yes/No) into numeric values

label_encoder = LabelEncoder()             # create encoder object
df['Pass'] = label_encoder.fit_transform(df['Pass'])   # fit + transform the text column

# Check here few rows again to confirm encoding is worked
print("\nEncoded dataset:")
print(df.head())

# Separate features and target, then train the Random Forest model

# Separate input columns (features) and output column (target)
X = df[['Hours_Study', 'Attendance', 'Assignments_Submitted']]
y = df['Pass']

# Create the Random Forest Classifier
model = RandomForestClassifier(n_estimators=10, random_state=1)

# Train the model using the data
model.fit(X, y)

# Confirming here training is completed
print("\nModel training completed.")


# Make predictions and test the model

# Predict results for the same dataset
predictions = model.predict(X)

# Predictions alongside the actual values
print("\nPredictions on training data:")
result = pd.DataFrame({'Actual': y, 'Predicted': predictions})
print(result.head())

# Test the model with a new student's data

new_student = pd.DataFrame([[5, 80, 3]], columns=['Hours_Study', 'Attendance', 'Assignments_Submitted'])
new_prediction = model.predict(new_student)

# Check prediction result
print("\nManual test input -> Hours_Study=5, Attendance=80, Assignments_Submitted=3")
print("Model Prediction:", "Pass" if new_prediction[0] == 1 else "Fail")


# Check which features are most important in making predictions

importance = model.feature_importances_

print("\nFeature Importance:")
for name, score in zip(X.columns, importance):
    print(f"{name}: {score:.3f}")
