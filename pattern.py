import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def extract_all_data(input_folder):
    """Reads and combines data from all CSVs in the input folder."""
    combined_data = pd.DataFrame()
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            try:
                # Read CSV files, skipping irregularities
                df = pd.read_csv(input_path, on_bad_lines='skip', delimiter=',', skiprows=2)
                combined_data = pd.concat([combined_data, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    print(f"Total rows extracted: {combined_data.shape[0]}")
    return combined_data


def preprocess_data(data):
    """Cleans and preprocesses the data for anomaly detection."""
    
    # Check how much data is left after dropping rows with any missing values
    print("Before dropping NaN values:", data.shape)
    
    # Only drop rows that are missing entirely in numerical columns, NOT the whole dataset
    data = data.dropna(subset=data.select_dtypes(include='number').columns, how='any')
    
    print("After dropping NaN values:", data.shape)

    # Normalize or scale features for ML models
    scaler = StandardScaler()
    
    # Select numerical columns dynamically for scaling
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    print("Numerical columns available for scaling:", numerical_cols)

    # Handle edge cases: Ensure we have numerical data to scale
    if numerical_cols.size == 0:
        print("No numerical data left for scaling. Exiting preprocessing.")
        return pd.DataFrame()
    
    try:
        scaled_data = scaler.fit_transform(data[numerical_cols])  # Scale only valid numerical columns
    except Exception as e:
        print("Error during scaling:", e)
        return pd.DataFrame()

    # Return the scaled data as a DataFrame
    return pd.DataFrame(scaled_data, columns=numerical_cols)


def detect_fault_patterns(data):
    """Applies anomaly detection to find fault patterns."""
    if data.empty:
        print("No valid data available for anomaly detection. Exiting.")
        return pd.DataFrame()
    
    # Using IsolationForest for anomaly detection
    try:
        model = IsolationForest(contamination=0.05, random_state=42)  # Contamination indicates the expected fraction of anomalies
        data['anomaly_score'] = model.fit_predict(data)
        # Map predictions (1 = normal, -1 = anomaly)
        data['is_fault'] = data['anomaly_score'].apply(lambda x: True if x == -1 else False)
    except Exception as e:
        print("Error during fault detection:", e)
        return pd.DataFrame()
    
    return data


def visualize_fault_patterns(data):
    """Visualize data to analyze anomalies."""
    # Avoid visualization if no anomalies are detected or data is invalid
    if 'is_fault' not in data.columns:
        print("No fault patterns detected. Skipping visualization.")
        return

    # Plotting a sample visualization of anomalies
    for column in data.select_dtypes(include='number').columns:  # Loop over numerical columns for visualization
        plt.figure(figsize=(12, 6))
        plt.scatter(data.index, data[column], c=data['is_fault'].map({True: 'red', False: 'blue'}), s=10)
        plt.xlabel("Index")
        plt.ylabel(column)
        plt.title(f"Anomalies detected in {column}")
        plt.show()


if __name__ == "__main__":
    # Define folder paths
    base_dir = os.getcwd()
    input_folder = os.path.join(base_dir, "raw_data")
    
    # Step 1: Extract all data from CSVs in the input directory
    print("Extracting data from CSVs...")
    data = extract_all_data(input_folder)
    
    # Step 2: Preprocess the data
    print("Preprocessing data...")
    preprocessed_data = preprocess_data(data)
    
    if preprocessed_data.empty:
        print("Preprocessing failed. Exiting.")
        exit()
    
    # Step 3: Detect anomalies
    print("Detecting fault patterns...")
    result_data = detect_fault_patterns(preprocessed_data)
    
    if result_data.empty:
        print("No anomalies were detected or preprocessing failed. Exiting.")
        exit()
    
    # Step 4: Visualize anomalies
    print("Visualizing anomalies...")
    visualize_fault_patterns(result_data)
    
    # Optional: Save the results to a CSV for further inspection
    result_data.to_csv('fault_analysis_results.csv', index=False)
    print("Results saved to 'fault_analysis_results.csv'")
