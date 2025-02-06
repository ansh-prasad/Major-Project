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

# # Directory where models and results will be saved
# # output_dir = r"D:\MAJOR\Major-Project\SUB- FOLDER\models_"
output_dir = r"models"
# os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists


# Load the CSV data
# data = pd.read_csv("Training_Data_prev_F100.csv")
data = pd.read_csv("Training_Data.csv")

# Ensure the 'Fault_label' column is treated as string and strip whitespace
data['Fault_label'] = data['Fault_label'].astype(str).str.strip()

# Encode the 'Fault_label' using LabelEncoder
label_encoder = LabelEncoder()
data['Fault_label'] = label_encoder.fit_transform(data['Fault_label'])

# Save the encoder for future use
joblib.dump(label_encoder, os.path.join(output_dir, "fault_label_encoder.pkl"))

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
    model_filename = f"{name.lower().replace(' ', '_')}_model.pkl"
    model_path = os.path.join(output_dir, model_filename)
    if os.path.exists(model_path):
        print(f"{name} model already exists. Skipping training.")
        # Load the existing model if you want to use it for predictions
        model = joblib.load(model_path)
    else:    
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

# Save combined results to a single CSV
output_path = os.path.join(output_dir, "all_models_predictions.csv")
combined_results.to_csv(output_path, index=False)
print(f"\nCombined results saved to '{output_path}'")

# Test on new data function
def predict_new_data(file_path):
    # Load new data
    new_data = pd.read_csv(file_path)
    
    has_actual_labels = 'Actual_Fault_Label' in new_data.columns
    
    if has_actual_labels:
        # Convert actual labels for comparison
        new_data['Actual_Fault_Label'] = new_data['Actual_Fault_Label'].astype(str).str.strip()
        new_data['Actual_Fault_Label'] = label_encoder.transform(new_data['Actual_Fault_Label'])
    

    # Preprocess the new data
    new_data_processed = new_data[features]  # Ensure columns match the training data
    
    # Load all trained models
    model_files = [f for f in os.listdir(output_dir) if f.endswith('_model.pkl')]
    
    # Initialize results DataFrame
    results = new_data.copy()
    
    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
        
        # Load the model
        model_path = os.path.join(output_dir, model_file)
        model = joblib.load(model_path)
        
        # Make predictions
        results[f'{model_name}_Pred_Fault_Label'] = model.predict(new_data_processed)
        results[f'{model_name}_Pred_Fault_Label'] = label_encoder.inverse_transform(results[f'{model_name}_Pred_Fault_Label'])
        proba = model.predict_proba(new_data_processed) 
        # Map probabilities for Normal (index 0), Warning (index 1), and Fault (index 2)
        prob_normal = proba[:, 0]*100  # Probability of Normal
        prob_warning = proba[:, 1]*200  # Probability of Warning
        prob_fault = proba[:, 2]*300  # Probability of Fault
        
        # Add predictions to the results DataFrame
        results[f'{model_name}_Pred_Fault_Label_proba_normal'] = prob_normal
        results[f'{model_name}_Pred_Fault_Label_proba_warning'] = prob_warning
        results[f'{model_name}_Pred_Fault_Label_proba_fault'] = prob_fault
        
        # If actual labels exist, calculate accuracy and classification report
        if has_actual_labels:
            accuracy = accuracy_score(new_data['Actual_Fault_Label'], model.predict(new_data_processed))
            print("for test data")
            print(f"\nModel: {model_name} - Accuracy: {accuracy:.8f}")
            print(f"Classification Report for {model_name}:")
            print(classification_report(new_data['Actual_Fault_Label'], model.predict(new_data_processed)))
    
    # Save predictions to a new file
    output_file = os.path.join(output_dir, "test_data_predictions.csv")
    results.to_csv(output_file, index=False)
    print(f"\nPredictions for all models saved to: {output_file}")
    
# Uncomment the line below to test new data with a specific model
predict_new_data("Filtered_Test_Data.csv")

test_predictions_file = os.path.join(output_dir, "Filtered_Test_Data.csv")

# Read the test predictions CSV
df = pd.read_csv(test_predictions_file)

# Add a Serial Number (S.No) column
df.insert(0, "S.No", range(1, len(df) + 1))

# Define a mapping for fault labels
fault_mapping = {'NORMAL': 100, 'WARNING': 200, 'FAULT': 300}

# Extract columns related to model predictions
# model_columns = [col for col in df.columns if '_Pred_Fault_Label' in col]
model_columns = [col for col in df.columns if '_Fault_Label' in col]
for col in model_columns:
    df[col] = df[col].astype(str).str.strip().str.upper()  # Remove spaces & standardize case
    df[col] = df[col].map(fault_mapping)  # Map categories to numbers

# Create a folder to save graphs
graphs_dir = os.path.join(output_dir, "fault_prediction_graphs")
os.makedirs(graphs_dir, exist_ok=True)

# Function to plot and save graphs
def plot_and_save_fault_graphs(df, model_columns, save_dir):
    """
    Plots and saves separate line graphs for each model's predicted fault labels.

    Args:
        df (DataFrame): The DataFrame containing predictions.
        model_columns (list): List of model prediction columns.
        save_dir (str): Directory to save the graphs.

    """
   
    for col in model_columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["S.No"], df[col], marker='o', linestyle='-', label=col.replace('_Pred_Fault_Label', ''))
        # Customize plot
        plt.xlabel("Row Index")
        plt.ylabel("Predicted Fault Label (100=Normal, 200=Warning, 300=Fault)")
        plt.title(f"Predicted Fault Labels - {col.replace('_Pred_Fault_Label', '')}")
        plt.yticks([100, 200, 300], ["Normal", "Warning", "Fault"])
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_filename = os.path.join(save_dir, f"{col.replace('_Pred_Fault_Label', '')}_fault_graph.png")
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to prevent overlapping plots

        print(f"Saved: {plot_filename}")

# Call the function to generate and save graphs
plot_and_save_fault_graphs(df, model_columns, graphs_dir)

"""-----------------------------------------------------------------------------------------------------------------------------------"""

test_predictions_file = os.path.join(output_dir, "test_data_predictions.csv")

# Read the test predictions CSV
df = pd.read_csv(test_predictions_file)

# Add a Serial Number (S.No) column
df.insert(0, "S.No", range(1, len(df) + 1))

# Define a mapping for fault labels
fault_mapping = {'NORMAL': 100, 'WARNING': 200, 'FAULT': 300}

model_columns = [col for col in df.columns if '_Pred_Fault_Label_proba_' in col]

def plot_and_save_fault_proba_graphs(df, model_columns, save_dir):
    """
    Plots and saves separate line graphs for each model's predicted fault labels.

    Args:
        df (DataFrame): The DataFrame containing predictions.
        model_columns (list): List of model prediction columns.
        save_dir (str): Directory to save the graphs.

    """
   
    for col in model_columns:
        plt.figure(figsize=(10, 5))
        model_name = col.split('_Pred_Fault_Label_proba_')[0]  # Extract model name from column
        # Separate the probability for 'Normal', 'Warning', and 'Fault'
        if 'normal' in col.lower():
            prob_column = 'NORMAL'
        elif 'warning' in col.lower():
            prob_column = 'WARNING'
        elif 'fault' in col.lower():
            prob_column = 'FAULT'
        else:
            continue
        
    
        # Customize plot
        plt.plot(df["S.No"], df[col], marker='o', linestyle='-', label=col.replace('_Pred_Fault_Label', ''))
        plt.xlabel("Row Index")
        plt.ylabel("Probability (%)")
        plt.title(f"{model_name} - {prob_column} Probability vs Serial Number")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_filename = os.path.join(save_dir, f"{model_name}_{prob_column}_probability_graph.png")
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to prevent overlapping plots

        print(f"Saved: {plot_filename}")

# Call the function to generate and save graphs
plot_and_save_fault_proba_graphs(df, model_columns, graphs_dir)