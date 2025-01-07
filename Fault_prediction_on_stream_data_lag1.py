import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import logging
import time
import random
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the processed data folder and model filename
PROCESSED_FOLDER = "processed_data"
MODEL_FILENAME = "solar_fault_predictive_model_lag.pkl"

# Simulate streaming data (will be read from the CSV file)
CSV_FILE = "test.csv"


# Function to generate random streaming data
def generate_random_data(stats):
    return {
        'Ambient Temp (C)': np.random.normal(stats['Ambient Temp (C)']['mean'], stats['Ambient Temp (C)']['std']),
        'Solar Radiation (W/m2)': np.random.normal(stats['Solar Radiation (W/m2)']['mean'], stats['Solar Radiation (W/m2)']['std']),
        'Load kW Sum (kW)': np.random.normal(stats['Load kW Sum (kW)']['mean'], stats['Load kW Sum (kW)']['std']),
    }


# Simulate streaming data
def simulate_streaming_data(csv_file: str, lag_length: int = 3):
    # Load the CSV file
    data = pd.read_csv(csv_file)
    
    if 'Description' not in data.columns:
        raise ValueError("The CSV file must include a 'Description' column for ground truth.")
    
    stats = {
        col: {'mean': data[col].mean(), 'std': data[col].std()}
        for col in ['Ambient Temp (C)', 'Solar Radiation (W/m2)', 'Load kW Sum (kW)']
    }
    
    # Initialize lag data
    lag_data = {col: [stats[col]['mean']] * lag_length for col in stats}
    correct_predictions = 0
    total_samples = 0
   
    
    for _  in data.iterrows():  # Iterate through the CSV rows to mimic streaming
        # Extract the current data and ground truth
        row = data.sample(n=1).iloc[0]
        
        
        current_data = row[['Ambient Temp (C)', 'Solar Radiation (W/m2)', 'Load kW Sum (kW)']].to_dict()
        actual_fault = row['Description']
        
        # Prepare features including lagged values
        features = current_data.copy()
        for i in range(1, lag_length + 1):
            for col in lag_data:
                features[f"{col}_lag{i}"] = lag_data[col][i - 1]
        
        # Update lag data
        for col in lag_data:
            lag_data[col] = [current_data[col]] + lag_data[col][:-1]
        
        # Create DataFrame for prediction
        new_data = pd.DataFrame([features])
        
        # Predict faults
        predicted_fault, fault_probabilities = predict_faults(new_data, MODEL_FILENAME)
        
        # Check correctness
        is_correct = predicted_fault == actual_fault
        correct_predictions += int(is_correct)
        total_samples += 1
        
        # Display results
        print(f"Streaming Row {total_samples}")
        print(f"Prediction: {predicted_fault}")
        print(f"Actual: {actual_fault}")
        print(f"Correct: {is_correct}")
        print("Probabilities:")
        for fault, probability in fault_probabilities.items():
            print(f"{fault}: {probability * 100:.2f}%")
        print("-" * 30)
        
        time.sleep(10)  # Simulate time delay
        
    # Display overall accuracy after streaming ends
    print(f"Overall Accuracy: {correct_predictions / total_samples * 100:.2f}%")

# Function to load and aggregate data
def load_data(folder: str) -> pd.DataFrame:
    all_data = []
    logging.info(f"Loading data from folder: {folder}")
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    data = pd.read_csv(file_path)
                    data['Source File'] = file  # Add source file info for context
                    data['Category'] = os.path.basename(os.path.dirname(root))  # Add category
                    all_data.append(data)
                except Exception as e:
                    logging.error(f"Failed to read {file_path}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Fault mapping
def map_faults(data: pd.DataFrame) -> pd.DataFrame:
    categories = {
        "FAULT DATA LOG": {
            "Fault : Ambient Temp Sensor Fault": "Ambient Temp Sensor Fault",
            "Fault : High DC Volts": "High DC Volts",
            "Fault : Inv Sync Fault": "Inv Sync Fault",
            "Fault : Spare Flt 6": "Spare Flt 6"
        },
        "STATUS DATA LOG": {
            "Status : Auto Inv Only Mode": "Auto Inv Only Mode",
            'Status : HMI Initialize ': 'HMI Initialize ',
            "Status : Battery Charge Stage 1": "Battery Charge Stage 1",
            "Status : Battery Charge Stage 2": "Battery Charge Stage 2",
            "Status : Battery Charge Stage 3": "Battery Charge Stage 3",
            "Status : Battery Charge Stage 4": "Battery Charge Stage 4",
            "Status : Inverter CB Closed": "Inverter CB Closed",
            "Status : Inverter CB Opened": "Inverter CB Opened",
            "Status : Inverter Masked": "Inverter Masked",
            "Status : Inverter Started": "Inverter Started",
            "Status : Inverter Stopped": "Inverter Stopped",
            "Status : Low Battery Bulk Enable": "Low Battery Bulk Enable",
            "Status : Parallel Mode": "Parallel Mode",
            "Status : Solar Control Disabled": "Solar Control Disabled",
            "Status : Solar Control Enabled": "Solar Control Enabled",
            "Status : Source A CB Closed": "Source A CB Closed",
            "Status : Source A CB Opened": "Source A CB Opened",
            "Status : Source A Cooling Down": "Source A Cooling Down",
            "Status : Source A Stopped": "Source A Stopped",
            "Status : Source A Warming Up": "Source A Warming Up",
            "Status : Start Source A": "Start Source A"
        },
        "USER DATA LOG": {
            "User : Fault Reset": "Fault Reset",
            "User : Inverter Started": "Inverter Started",
            "User : Remote GSM connected": "Remote GSM connected",
            "User : Remote GSM disconnected": "Remote GSM disconnected",
            "User : System Off": "System Off"
        },"NORMAL DATA LOG": {
            "Data Log": "Normal Data"
        }
        
   }
    fault_map = {desc: category for category_dict in categories.values() for desc, category in category_dict.items()}
    data['Description'] = data['Description'].map(fault_map)
    
    # Log unmapped descriptions for troubleshooting
    unmapped = data[data['Description'].isna()]['Description'].unique()
    if len(unmapped) > 0:
        logging.warning(f"Unmapped descriptions found: {unmapped}")
    
    return data

# Function to train and save the model
def train_model(data: pd.DataFrame):
    logging.info("Training model...")
    features = ['Ambient Temp (C)', 'Solar Radiation (W/m2)', 'Load kW Sum (kW)'] + \
               [f"{col}_lag{lag}" for col in ['Ambient Temp (C)', 'Solar Radiation (W/m2)', 'Load kW Sum (kW)'] for lag in range(1, 4)]
    
    # Drop rows with NaN fault labels
    data = data.dropna(subset=['Description'])
    if data.empty:
        logging.error("No valid data with fault labels available for training.")
        exit()

    X = data[features]
    y = data['Description']

    # # Handle class imbalance with SMOTE
    # smote = SMOTE(random_state=42, k_neighbors=3)  # Set k_neighbors to 3 to avoid the error
    # X, y = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    logging.info("Model evaluation results:")
    print(classification_report(y_test, y_pred, zero_division=1))

    # Save the trained model
    joblib.dump(model, MODEL_FILENAME)
    logging.info(f"Model saved as {MODEL_FILENAME}")

# Function to predict faults
def predict_faults(new_data: pd.DataFrame, model_filename: str) -> tuple:
    logging.info("Predicting faults...")
    
    # Load the trained model
    model = joblib.load(model_filename)
    
    # Ensure the features match the model
    required_features = model.feature_names_in_
    missing_features = set(required_features) - set(new_data.columns)
    
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Reorder columns to match the model's expected input
    new_data = new_data[required_features]
    
    # Predict probabilities
    probabilities = model.predict_proba(new_data)
    class_probabilities = dict(zip(model.classes_, probabilities[0]))
    predicted_class = model.classes_[probabilities.argmax(axis=1)][0]
    
    return predicted_class, class_probabilities

# Main script
if __name__ == "__main__":
    # Load and preprocess data
    data = load_data(PROCESSED_FOLDER)
    if data.empty:
        logging.error("No data loaded. Please check the processed folder.")
        exit()

    # Ensure necessary columns are present
    required_columns = {'Description', 'Ambient Temp (C)', 'Solar Radiation (W/m2)', 'Load kW Sum (kW)'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        logging.error(f"Required columns are missing: {missing}")
        exit()

    # Map faults
    data = map_faults(data)

    # Check if the model file exists
    if os.path.exists(MODEL_FILENAME):
        print(f"Loading existing model from {MODEL_FILENAME}...")
        model = joblib.load(MODEL_FILENAME)
    else:
        print("Model not found. Training a new model...")
        # Load data (assuming you have a function to load your data)
        data = load_data("processed_data")
        # Train and save model
        train_model(data)


try:
        for predicted_fault in simulate_streaming_data(CSV_FILE):
            # Display the result (output will be logged by logging.info)
            pass
except KeyboardInterrupt:
        print("\nStreaming interrupted.")
    # # Example prediction   
    # new_sample = pd.DataFrame([{
    #     'Ambient Temp (C)': 33.6,
    #     'Solar Radiation (W/m2)': 863,
    #     'Load kW Sum (kW)': 1.8,
    #     'Ambient Temp (C)_lag1': 33.7,
    #     'Solar Radiation (W/m2)_lag1': 868,
    #     'Load kW Sum (kW)_lag1': 0.8,
    #     'Ambient Temp (C)_lag2': 31.9,
    #     'Solar Radiation (W/m2)_lag2': 208,
    #     'Load kW Sum (kW)_lag2': 3.4,
    #     'Ambient Temp (C)_lag3':  32.8,
    #     'Solar Radiation (W/m2)_lag3': 489,
    #     'Load kW Sum (kW)_lag3': 9.1,
    # }])

    # # Align features and predict
    # predicted_fault, fault_probabilities = predict_faults(new_sample, MODEL_FILENAME)

    # print("-------------------------")
    # print(f"Prediction: {predicted_fault}")
    # print("-------------------------")
    # print("Prediction Probabilities:")
    # print("-------------------------")
    # for fault, probability in fault_probabilities.items():
    #     if probability > 0:
    #         print(f"{fault}: {probability * 100:.2f}%")
