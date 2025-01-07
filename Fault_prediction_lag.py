import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define constants
PROCESSED_FOLDER = "processed_data"
MODEL_FILENAME = "solar_fault_predictive_model_lag.pkl"

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
                    all_data.append(data)
                except Exception as e:
                    logging.error(f"Failed to read {file_path}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Function to add time-lagged features
def add_time_lagged_features(data: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    logging.info("Adding time-lagged features...")
    for lag in range(1, lags + 1):
        for col in ['Ambient Temp (C)', 'Solar Radiation (W/m2)', 'Load kW Sum (kW)']:
            data[f"{col}_lag{lag}"] = data[col].shift(lag)
    return data.dropna()

# Fault mapping function
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
        },
        "NORMAL DATA LOG": {
            "Data Log": "Normal Data"
        }
    }
    fault_map = {desc: category for category_dict in categories.values() for desc, category in category_dict.items()}
    data['Fault_Label'] = data['Description'].map(fault_map)
    data['Fault_Label'] = data['Fault_Label'].fillna(data['Description'])  # Retain unmapped descriptions
    return data

# Function to train and save the model
def train_model(data: pd.DataFrame):
    logging.info("Training model...")
    features = ['Ambient Temp (C)', 'Solar Radiation (W/m2)', 'Load kW Sum (kW)'] + \
               [f"{col}_lag{lag}" for col in ['Ambient Temp (C)', 'Solar Radiation (W/m2)', 'Load kW Sum (kW)'] for lag in range(1, 4)]

    # Replace missing fault labels with "Unknown" rather than dropping them
    data['Fault_Label'] = data['Fault_Label'].fillna("Unknown")

    # Split data into features (X) and target (y)
    X = data[features]
    y = data['Fault_Label']

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

    # Map faults and add time-lagged features
    data = map_faults(data)
    data = add_time_lagged_features(data)

    # Check if the model file exists
    if os.path.exists(MODEL_FILENAME):
        logging.info(f"Loading existing model from {MODEL_FILENAME}...")
        model = joblib.load(MODEL_FILENAME)
    else:
        logging.info("Model not found. Training a new model...")
        train_model(data)
