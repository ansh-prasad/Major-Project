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
                data['Category'] = os.path.basename(os.path.dirname(root))  # Add category (e.g., FAULT, STATUS)
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

# Simulate live data prediction
# def predict_faults(new_data):
#     probabilities = model.predict_proba(new_data)
#     predicted_class_index = probabilities.argmax(axis=1)[0]
#     predicted_class = model.classes_[predicted_class_index]
#     predicted_probability = probabilities[0][predicted_class_index] * 100
#     return predicted_class, predicted_probability

def predict_faults(new_data):
    probabilities = model.predict_proba(new_data)
    class_probabilities = dict(zip(model.classes_, probabilities[0]))  # Map classes to their probabilities
    predicted_class = model.classes_[probabilities.argmax(axis=1)][0]
    return predicted_class, class_probabilities



# Example of prediction
new_sample = pd.DataFrame([{
    'Ambient Temp (C)': 37,
    'Solar Radiation (W/m2)': 900,
    'Load kW Sum (kW)': 0,
    'Inv kW Sum (kW)': 0,
    'Solar kW (kW)': 0.3
}])

# predictions, probability = predict_faults(new_sample)
# print(f"Predicted Fault: {predictions}")
# print(f"Fault Probability: {probability:.2f}%")

predicted_fault, fault_probabilities = predict_faults(new_sample)
print(f"Prediction : {predicted_fault}")
print("-------------------------")
print("Prediction Probabilities:")
print("-------------------------")
for fault, probability in fault_probabilities.items():
    if probability > 0:  # Only print categories with probability greater than 0
        print(f"{fault}: {probability*100:.2f}%")