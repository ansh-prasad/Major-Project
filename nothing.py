import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Folder containing the training CSV files
train_folder_path = "/Users/ansh/Music/AP/AP_Code/Major-Project/modified_raw_data"  # Replace with your training folder path

# File containing the testing CSV data
test_file_path = "/Users/ansh/Music/AP/AP_Code/Major-Project/test.csv"  # Replace with your testing file path

# Function to load and combine all CSV files
def load_data_from_folder(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = [pd.read_csv(file) for file in all_files]
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data

# Function to label categories
def label_category(description):
    if isinstance(description, str):  # Check if description is a string
        if "Fault" in description:
            return "FAULT"
        elif "Status" in description:
            return "STATUS"
        elif "User" in description:
            return "USER"
        else:
            return "NORMAL"
    return "UNKNOWN"  # For missing or invalid descriptions

# Load and preprocess training data
train_data = load_data_from_folder(train_folder_path)
train_data['Category'] = train_data['Description'].apply(label_category)

# Select numeric features for training
features = [
    "Inv kW Sum (kW)", "Load kW Sum (kW)", "Solar kW (kW)",
    "Ambient Temp (C)", "Solar Radiation (W/m2)", "Inv Exp kWh",
    "Inv Imp kWh", "Src A Exp kWh", "Src A Imp kWh", "Src B Exp kWh",
    "Src B Imp kWh", "Site kWh (calc)", "Batt Exp kWh", "Batt Imp kWh", "Solar kWh"
]
X_train = train_data[features]
y_train = train_data['Category']

# Handle missing values in training data
X_train.fillna(X_train.median(), inplace=True)

# Encode target variable
y_train_encoded = y_train.map({"NORMAL": 0, "FAULT": 1, "STATUS": 2, "USER": 3, "UNKNOWN": 4})

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_encoded)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train_resampled)

# Load and preprocess testing data
test_data = pd.read_csv(test_file_path)
test_data['Category'] = test_data['Description'].apply(label_category)

X_test = test_data[features]
y_test = test_data['Category']

# Handle missing values in testing data
X_test.fillna(X_test.median(), inplace=True)

# Encode test target variable
y_test_encoded = y_test.map({"NORMAL": 0, "FAULT": 1, "STATUS": 2, "USER": 3, "UNKNOWN": 4})

# Scale testing data
X_test_scaled = scaler.transform(X_test)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Map encoded predictions back to categories
category_map = {0: "NORMAL", 1: "FAULT", 2: "STATUS", 3: "USER", 4: "UNKNOWN"}
y_test_categories = y_test_encoded.map(category_map)
y_pred_categories = pd.Series(y_pred).map(category_map)

# Print predictions alongside actual categories
comparison = pd.DataFrame({
    "Description": test_data["Description"],
    "Actual": y_test_categories.values,
    "Predicted": y_pred_categories.values
})
print("Comparison of Actual and Predicted Categories:\n", comparison.head(20))  # Display first 20 rows

# Evaluate the model
print("Classification Report:\n", classification_report(y_test_encoded, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred))
