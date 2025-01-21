import os
import pandas as pd

def extract_columns(input_folder, output_folder, required_columns):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each CSV file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):  # Process only .csv files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Read the CSV file with error handling for irregular rows and skip first 2 rows
                df = pd.read_csv(input_path, on_bad_lines='skip', delimiter=',', skiprows=2)

                # Ensure all required columns are in the DataFrame, filling missing ones with NaN
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = None

                # Reorder the DataFrame to match the required_columns order
                df = df[required_columns]

                # Save the modified DataFrame to the output folder
                df.to_csv(output_path, index=False)

                print(f"Processed and saved: {filename}")
            except Exception as e:
                # Log error
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Define input and output folder paths
    base = os.getcwd()  # Base directory
    input_folder = os.path.join(base, "test-1")  # Input folder path
    output_folder = os.path.join(base, "modified_test-1")  # Output folder path

    # Define the required columns based on your note
    required_columns = [
        "Date", "Time", "Description", "Trigger", "Inv kW Sum (kW)", "Load kW Sum (kW)", "Solar kW (kW)", "Ambient Temp (C)",
        "Solar Radiation (W/m2)", "Inv Exp kWh", "Inv Imp kWh", "Src A Exp kWh", "Src A Imp kWh", "Src B Exp kWh",
        "Src B Imp kWh", "Site kWh (calc)", "Batt Exp kWh", "Batt Imp kWh", "Solar kWh"
    ]

    # Execute extraction
    extract_columns(input_folder, output_folder, required_columns)
