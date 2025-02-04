import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Define the processed data folder
processed_folder = "processed_data"

# Aggregate data
def load_data(folder):
    all_data = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path)
                data['Source File'] = file  # Add source file info for context
                data['Category'] = os.path.basename(os.path.dirname(root))  # Add category (e.g., FAULT, WARNING)
                all_data.append(data)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Load and preprocess data
data = load_data(processed_folder)

# Ensure necessary columns are present
if not {'Description', 'Ambient Temp (C)', 'Solar Radiation (W/m2)', 'Load kW Sum (kW)'}.issubset(data.columns):
    raise ValueError("Required columns are missing in the data.")

# Category structure for mapping
categories = {
    "FAULT DATA LOG": {
        "Fault : Ambient Temp Sensor Fault": "Ambient Temp Sensor Fault",
        "Fault : High DC Volts": "High DC Volts",
        "Fault : Inv Sync Fault": "Inv Sync Fault",
        "Fault : Spare Flt 6": "Spare Flt 6"
    },
    "WARNING DATA LOG": {
        "WARNING : Auto Inv Only Mode": "Auto Inv Only Mode",
        "WARNING : Battery Charge Stage 1": "Battery Charge Stage 1",
        "WARNING : Battery Charge Stage 2": "Battery Charge Stage 2",
        "WARNING : Battery Charge Stage 3": "Battery Charge Stage 3",
        "WARNING : Battery Charge Stage 4": "Battery Charge Stage 4",
        "WARNING : Inverter CB Closed": "Inverter CB Closed",
        "WARNING : Inverter CB Opened": "Inverter CB Opened",
        "WARNING : Inverter Masked": "Inverter Masked",
        "WARNING : Inverter Started": "Inverter Started",
        "WARNING : Inverter Stopped": "Inverter Stopped",
        "WARNING : Low Battery Bulk Enable": "Low Battery Bulk Enable",
        "WARNING : Parallel Mode": "Parallel Mode",
        "WARNING : Solar Control Disabled": "Solar Control Disabled",
        "WARNING : Solar Control Enabled": "Solar Control Enabled",
        "WARNING : Source A CB Closed": "Source A CB Closed",
        "WARNING : Source A CB Opened": "Source A CB Opened",
        "WARNING : Source A Cooling Down": "Source A Cooling Down",
        "WARNING : Source A Stopped": "Source A Stopped",
        "WARNING : Source A Warming Up": "Source A Warming Up",
        "WARNING : Start Source A": "Start Source A",
        "WARNING : Fault Reset": "Fault Reset",
        "WARNING : Inverter Started": "Inverter Started",
        "WARNING : Remote GSM connected": "Remote GSM connected",
        "WARNING : Remote GSM disconnected": "Remote GSM disconnected",
        "WARNING : System Off": "System Off"
    },
    
    "NORMAL DATA LOG": {
        "Data Log": "Normal Data"
    }
}

# Map descriptions to fault labels
fault_map = {desc: category for category_dict in categories.values() for desc, category in category_dict.items()}
data['Fault_Label'] = data['Description'].map(fault_map)

# Filter relevant features and drop rows with NaN faults
features = ['Ambient Temp (C)', 'Solar Radiation (W/m2)', 'Load kW Sum (kW)', 'Inv kW Sum (kW)', 'Solar kW (kW)']
data = data.dropna(subset=['Fault_Label'])  # Only keep rows with fault labels

# Split data into features (X) and target (y)
X = data[features]
y = data['Fault_Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'solar_fault_model.pkl')

# Load the model for prediction
model = joblib.load('solar_fault_model.pkl')

# Function to predict faults with aggregated probabilities
def predict_faults(new_data):
    probabilities = model.predict_proba(new_data)
    class_probabilities = dict(zip(model.classes_, probabilities[0]))  # Map classes to probabilities
    predicted_class = model.classes_[probabilities.argmax(axis=1)][0]

    # Define category mappings
    category_mapping = {
        "FAULT": ["Ambient Temp Sensor Fault", "High DC Volts", "Inv Sync Fault", "Spare Flt 6"],
        "WARNING": ["Auto Inv Only Mode", "Battery Charge Stage 1", "Battery Charge Stage 2",
                    "Battery Charge Stage 3", "Battery Charge Stage 4", "Inverter CB Closed",
                    "Inverter CB Opened", "Inverter Masked", "Inverter Started", "Inverter Stopped",
                    "Low Battery Bulk Enable", "Parallel Mode", "Solar Control Disabled",
                    "Solar Control Enabled", "Source A CB Closed", "Source A CB Opened",
                    "Source A Cooling Down", "Source A Stopped", "Source A Warming Up", "Start Source A , Fault Reset", "Inverter Started", "Remote GSM connected", "Remote GSM disconnected", "System Off"],
        
        "NORMAL": ["Normal Data"]
    }

    # Aggregate probabilities into broader categories
    aggregated_probabilities = {"FAULT": 0, "WARNING": 0,  "NORMAL": 0}
    for category, faults in category_mapping.items():
        aggregated_probabilities[category] = sum(prob for fault, prob in class_probabilities.items() if fault in faults)

    return predicted_class, aggregated_probabilities

# Example of prediction
new_sample = pd.DataFrame([{
    'Ambient Temp (C)': 36.1,
    'Solar Radiation (W/m2)': 30,
    'Load kW Sum (kW)': 0.8,
    'Inv kW Sum (kW)': -0.3,
    'Solar kW (kW)': 0
}])

predicted_fault, fault_probabilities = predict_faults(new_sample)
print(f"Prediction: {predicted_fault}")
print("-------------------------")
print("Prediction Probabilities:")
print("-------------------------")
for category, probability in fault_probabilities.items():
    print(f"{category}: {probability*100:.2f}%")
