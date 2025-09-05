import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import missingno as msno 

import joblib

from sklearn.model_selection import RandomizedSearchCV, train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,mean_absolute_error,mean_squared_error,r2_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor

import folium
from folium.plugins import HeatMap
import plotly.express as px
# Load the dataset first to implement the model
data = pd.read_csv(r"C:\Users\sarth\Downloads\StudentPerformanceFactors.csv")
#To find the column which has null values in the dataset
print(data.isnull().sum())

#Storing the missing values columns under one name to effectly access them
missing_columns = ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']

#Runnning a for loop for the missing values in columns and fill them with mode values
for col in missing_columns:
    data[col].fillna(data[col].mode()[0],inplace=True)

from sklearn.preprocessing import LabelEncoder
#List of attributes whose values can be 
binary_columns = ['Internet_Access','Learning_Disabilities','Extracurricular_Activities']

for col in binary_columns:
    data[col] = data[col].map({'Yes':1,'No':0})
    
categorical_features = [
    'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
    'Family_Income', 'Teacher_Quality', 'Peer_Influence',
    'Parental_Education_Level', 'Distance_from_Home'
]

# Create a dictionary of encoders
encoder_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoder_dict[col] = le

# Save the encoder dictionary
joblib.dump(encoder_dict, "encoder.joblib")
    
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of Exam Scores
plt.figure(figsize=(8, 5))
sns.histplot(data['Exam_Score'], bins=30, kde=True, color="blue")
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Frequency")
plt.show()


# Scatter plot of Hours Studied vs Exam Score
plt.figure(figsize=(8, 5))
sns.scatterplot(x= data['Hours_Studied'], y= data['Exam_Score'], color="green")
plt.title("Hours Studied vs. Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.show()


# Boxplot of Tutoring Sessions vs Exam Score
plt.figure(figsize=(8, 5))
sns.boxplot(x= data['Tutoring_Sessions'], y= data['Exam_Score'], palette="coolwarm")
plt.title("Effect of Tutoring Sessions on Exam Scores")
plt.xlabel("Number of Tutoring Sessions")
plt.ylabel("Exam Score")
plt.show()

# Boxplot of Motivation Level vs Exam Score
plt.figure(figsize=(8, 5))
sns.boxplot(x= data['Motivation_Level'], y=data['Exam_Score'], palette="magma")
plt.title("Impact of Motivation Level on Exam Scores")
plt.xlabel("Motivation Level")
plt.ylabel("Exam Score")
plt.show()

# List of low-importance features
low_importance_features = ['School_Type', 'Learning_Disabilities', 'Gender', 'Extracurricular_Activities', 'Internet_Access']

# Filter out columns that don't exist in the DataFrame
existing_columns_to_drop = [col for col in low_importance_features if col in data.columns]

# Drop the existing columns
data = data.drop(columns=existing_columns_to_drop)



# Create new meaningful features
data['Study_Efficiency'] = data['Hours_Studied'] / (data['Attendance'] + 1)  # Avoid division by zero
data['Improvement_Rate'] = data['Exam_Score'] / (data['Hours_Studied']+1)  # How much they improved
data['Tutoring_Effect'] = data['Tutoring_Sessions'] / (data['Hours_Studied'] + 1)

print(data[['Study_Efficiency','Improvement_Rate','Tutoring_Effect']].head())


from sklearn.preprocessing import StandardScaler

# List of numerical columns to scale
numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                      'Tutoring_Sessions', 'Physical_Activity', 'Study_Efficiency', 
                      'Improvement_Rate', 'Tutoring_Effect']

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform numerical features
data[numerical_features] = scaler.fit_transform(data[numerical_features])


from sklearn.model_selection import train_test_split

print(data.head(10))

# Define target variable (Exam_Score) and features (X)
X = data.drop(columns=['Exam_Score'])  # Features
y = data['Exam_Score']  # Target variable

print("🚀 Features used for training:", list(X.columns))
joblib.dump(list(X.columns), "feature_names.joblib")  # Save feature names


# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting with default parameters
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Predict on test data
y_pred = gb_model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print model evaluation results
print("\n🚀 Gradient Boosting Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


# Define the model
gb_model = GradientBoostingRegressor(random_state=42)

# Define optimized hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300],  # Number of boosting stages
    'learning_rate': [0.05, 0.1, 0.15],  # Learning rate options
    'max_depth': [3, 5, 7],  # Tree depth options
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required in a leaf node
}

# Use RandomizedSearchCV for faster tuning
random_search = RandomizedSearchCV(
    gb_model, param_distributions=param_dist,
    n_iter=10,  # Search 10 random combinations
    cv=3,  # 3-fold cross-validation
    scoring='r2',
    n_jobs=-1, verbose=1, random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Get best model parameters
best_params = random_search.best_params_
print("\n✅ Best Parameters for Gradient Boosting:", best_params)

# Train the final optimized model
best_gb_model = GradientBoostingRegressor(**best_params, random_state=42)
best_gb_model.fit(X_train, y_train)

# Evaluate the optimized model
y_pred = best_gb_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print final performance
print("\n🚀 Optimized Gradient Boosting Performance:")
print(f"Mean Absolute Error(MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

joblib.dump(best_gb_model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")