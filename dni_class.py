''' in this new dni _class commit we along side created lag features'''

import os
import pandas as pd

def process_files(input_folder, max_lags= 3):
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(input_folder, filename)

            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Ensure the required columns are present
                required_columns = [
                    "Date", "Time", "Description", "Trigger", "Inv kW Sum (kW)", "Load kW Sum (kW)", 
                    "Solar kW (kW)", "Ambient Temp (C)", "Solar Radiation (W/m2)", "Inv Exp kWh", 
                    "Inv Imp kWh", "Src A Exp kWh", "Src A Imp kWh", "Src B Exp kWh", "Src B Imp kWh", 
                    "Site kWh (calc)", "Batt Exp kWh", "Batt Imp kWh", "Solar kWh"
                ]
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing columns in {filename}: {missing_columns}")

                # Add DNI column: Solar Radiation (W/m2) * (10 minutes converted to hours)
                df['DNI'] = df['Solar Radiation (W/m2)'] * (1 / 6)

                # Add norm DNI column: normalize DNI by the maximum DNI value
                max_dni = df['DNI'].max()
                if max_dni > 0:
                    df['norm DNI'] = df['DNI'] / max_dni
                else:
                    df['norm DNI'] = 0

                # Add calls norm DNI column: classify norm DNI values into categories
                def classify_dni(norm_dni):
                    if norm_dni <= 0.3:
                        return "Inspecting"
                    elif norm_dni <= 0.7:
                        return "Monitoring"
                    else:
                        return "Running"

                df['calls norm DNI'] = df['norm DNI'].apply(classify_dni)

                # Create lag features for specified columns
                lag_columns = [
                    "Inv kW Sum (kW)", "Load kW Sum (kW)", "Solar kW (kW)", "Ambient Temp (C)", 
                    "Solar Radiation (W/m2)", "Inv Exp kWh", "Inv Imp kWh", "Src A Exp kWh", 
                    "Src A Imp kWh", "Src B Exp kWh", "Src B Imp kWh", "Site kWh (calc)", 
                    "Batt Exp kWh", "Batt Imp kWh", "Solar kWh"
                ]

                for col in lag_columns:
                    for lag in range(1, max_lags + 1):
                        lag_col_name = f"{col}_lag{lag}"
                        df[lag_col_name] = df[col].shift(lag)

                # Save the updated file back to the same folder
                df.to_csv(file_path, index=False)

                print(f"Processed and updated: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Define the folder containing the modified raw data
    input_folder = os.path.join(os.getcwd(), "modified_raw_data")

    # Process the files with up to 3 lags
    process_files(input_folder, max_lags=3)
