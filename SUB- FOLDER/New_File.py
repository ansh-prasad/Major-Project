import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Directory where models and results will be saved
output_dir = r"models"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Load the CSV data
data = pd.read_csv("Training_Data.csv")

# Ensure the 'Fault_label' column is treated as string and strip whitespace
data['Fault_label'] = data['Fault_label'].astype(str).str.strip()

# Encode the 'Fault_label' using LabelEncoder
label_encoder = LabelEncoder()
data['Fault_label'] = label_encoder.fit_transform(data['Fault_label'])

# Save the encoder for future use
joblib.dump(label_encoder, os.path.join(output_dir, "fault_label_encoder.pkl"))

# Define the mapping for fault labels
fault_mapping = {"NORMAL": 1, "WARNING": 2, "FAULT": 3}

# Preprocessing: Select relevant features and the target column
features = ['Inv kW Sum (kW)', 'Load kW Sum (kW)', 'Solar kW (kW)', 'Ambient Temp (C)', 'Solar Radiation (W/m2)', 
            'Inv Exp kWh', 'Inv Imp kWh', 'Src A Exp kWh', 'Src A Imp kWh', 'Src B Exp kWh', 'Src B Imp kWh', 
            'Site kWh (calc)', 'Batt Exp kWh', 'Batt Imp kWh', 'Solar kWh']
X = data[features]
y = data['Fault_label']

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of models to train and evaluate
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB()
}

# Initialize a DataFrame to store all predictions
combined_results = X_test.copy()
combined_results['Actual_Fault_Label'] = y_test.values
combined_results['Actual_Fault_Label'] = label_encoder.inverse_transform(combined_results['Actual_Fault_Label'])

# Train, evaluate and save each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Save the trained model
    model_path = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
    joblib.dump(model, model_path)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Add predictions to the combined results DataFrame
    combined_results[f'{name}_Pred_Fault_Label'] = y_pred
    combined_results[f'{name}_Pred_Fault_Label'] = label_encoder.inverse_transform(combined_results[f'{name}_Pred_Fault_Label'])

# Add numerical classification column
combined_results['Fault_Label_Numeric'] = combined_results['Actual_Fault_Label'].map(fault_mapping)

# Save combined results to a single CSV
output_path = os.path.join(output_dir, "all_models_predictions.csv")
combined_results.to_csv(output_path, index=False)
print(f"\nCombined results saved to '{output_path}'")

# Function to predict new data
def predict_new_data(file_path):
    # Load new data
    new_data = pd.read_csv(file_path)
    has_actual_labels = 'Actual_Fault_Label' in new_data.columns
    
    if has_actual_labels:
        new_data['Actual_Fault_Label'] = new_data['Actual_Fault_Label'].astype(str).str.strip()
        new_data['Actual_Fault_Label'] = label_encoder.transform(new_data['Actual_Fault_Label'])
    
    # Preprocess the new data
    new_data_processed = new_data[features]
    
    # Load all trained models
    model_files = [f for f in os.listdir(output_dir) if f.endswith('_model.pkl')]
    results = new_data.copy()
    
    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
        model_path = os.path.join(output_dir, model_file)
        model = joblib.load(model_path)
        
        # Make predictions
        results[f'{model_name}_Pred_Fault_Label'] = model.predict(new_data_processed)
        results[f'{model_name}_Pred_Fault_Label'] = label_encoder.inverse_transform(results[f'{model_name}_Pred_Fault_Label'])
        
        # If actual labels exist, calculate accuracy
        if has_actual_labels:
            accuracy = accuracy_score(new_data['Actual_Fault_Label'], model.predict(new_data_processed))
            print(f"\nModel: {model_name} - Accuracy: {accuracy:.8f}")
            print(f"Classification Report for {model_name}:")
            print(classification_report(new_data['Actual_Fault_Label'], model.predict(new_data_processed)))
    
    # Add numerical classification column
    results['Fault_Label_Numeric'] = results['Actual_Fault_Label'].map(fault_mapping)
    
    # Save predictions
    output_file = os.path.join(output_dir, "test_data_predictions.csv")
    results.to_csv(output_file, index=False)
    print(f"\nPredictions for all models saved to: {output_file}")

# Uncomment the line below to test new data
predict_new_data("Test_Data_1_prev.csv")
