import os
import pandas as pd

def extract_columns(input_folder, output_folder, required_columns):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define mappings for Fault_label
    
    FAULT = {
        "Fault : Ambient Temp Sensor Fault",
        "Fault : High DC Volts",
        "Fault : Inv Sync Fault",
        "Fault : Spare Flt 6"
    }
    WARNING = {
        "Status : Auto Inv Only Mode",
        "Status : Battery Charge Stage 1",
        "Status : Battery Charge Stage 2",
        "Status : Battery Charge Stage 3",
        "Status : Battery Charge Stage 4",
        "Status : Inverter CB Closed",
        "Status : Inverter CB Opened",
        "Status : Inverter Masked",
        "Status : Inverter Started",
        "Status : Inverter Stopped",
        "Status : Low Battery Bulk Enable",
        "Status : Parallel Mode",
        "Status : Solar Control Disabled",
        "Status : Solar Control Enabled",
        "Status : Source A CB Closed",
        "Status : Source A CB Opened",
        "Status : Source A Cooling Down",
        "Status : Source A Stopped",
        "Status : Source A Warming Up",
        "Status : Start Source A",
        "User : Fault Reset",
        "User : Inverter Started",
        "User : Remote GSM connected",
        "User : Remote GSM disconnected",
        "User : System Off"
    },
    NORMAL = {
        "Data Log": "Data Log"
    }
    
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
                
                
                # Add the Fault_label column based on Description
                def assign_fault_label(description):
                    if not isinstance(description, str):
                     description = str(description)

                    description = description.strip()
                    if description in NORMAL:
                        return "NORMAL"
                    # elif description in WARNING:
                    #     return "WARNING"
                    elif description in FAULT:
                        return "FAULT"
                    else:
                        return "WARNING"

                df['Fault_label'] = df['Description'].apply(assign_fault_label)
             
                # # Propagate 'FAULT' to the previous 20 rows for any "FAULT" events
                # for i in range(1, len(df)):
                #     if "FAULT" in df.loc[i, 'Fault_label']:
                #         for j in range(1, 101):  # Propagate to the previous 4 rows
                #             if i - j >= 0:
                #                 df.loc[i - j, 'Fault_label'] = "FAULT"
                #                 # df.loc[i - j, 'Fault_label'] = "UNKNOWN"
                
                N = 20
                # Find indices where 'Fault_label' is "FAULT"
                fault_indices = df[df['Fault_label'] == "FAULT"].index.tolist()
                
                # Propagate FAULT label to previous n rows
                for idx in fault_indices:
                    for j in range(1, N + 1):
                        if idx - j >= 0:
                            df.loc[idx - j, 'Fault_label'] = "FAULT"
                                
                # Remove the original fault rows
                df = df.drop(index=fault_indices).reset_index(drop=True)
                
                
                
                # Propagate 'FAULT' to the next 20 rows for any "FAULT" events
                # for i in range(len(df)):
                #     if "FAULT" in df.loc[i, 'Fault_label']:
                #         for j in range(1, 6):  # Propagate to the next 20 rows
                #             if i + j < len(df):
                #                 df.loc[i + j, 'Fault_label'] = "FAULT"
                #                 # df.loc[i + j, 'Fault_label'] = "UNKNOWN"
                # Step 1: Identify and store indices where "FAULT" occurs
                # fault_indices = [i for i in range(len(df)) if df.loc[i, 'Fault_label'] == "FAULT"]
                
                # # Step 2: Propagate "FAULT" to the next 5 rows for each noted index
                # for i in fault_indices:
                #     for j in range(1, 21):  # Propagate to the next 5 rows
                #         if i + j < len(df):  # Ensure the index is within bounds
                #             df.loc[i + j, 'Fault_label'] = "FAULT"
                
                # Adjust Fault_label for 2 rows above and below if "Fault" is detected
                # fault_indices = df[df['Fault_label'] == "FAULT"].index

                # for idx in fault_indices:
                #     for offset in range(-2, 3):  # Range to include 2 rows above and 2 rows below
                #         adjusted_idx = idx + offset
                #         if 0 <= adjusted_idx < len(df):  # Ensure index is within bounds
                #             df.loc[adjusted_idx, 'Fault_label'] = "FAULT"
                
                # # between unknown fault label
                # for idx in fault_indices:
                #     # Ensure the actual fault retains its label as "FAULT"
                #     df.loc[idx, 'Fault_label'] = "FAULT"
                    
                #     # Update two rows above and two rows below
                #     for offset in range(-10, 11):  # Include 2 rows above and 2 rows below
                #         adjusted_idx = idx + offset
                #         if adjusted_idx != idx and 0 <= adjusted_idx < len(df):  # Skip the actual fault index and ensure bounds
                #             df.loc[adjusted_idx, 'Fault_label'] = "UNKNOWN"
                            
                            
                # Reorder the DataFrame to match the required_columns order and include Fault_label
                if "Fault_label" not in required_columns:
                    required_columns.append("Fault_label")
                df = df[required_columns] 
                
                # Reorder the DataFrame to match the required_columns order
                # df = df[required_columns]

                # Save the modified DataFrame to the output folder
                df.to_csv(output_path, index=False)

                print(f"Processed and saved: {filename}")
            except Exception as e:
                # Log error
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Define input and output folder paths
    base = os.getcwd()  # Base directory
    input_folder = os.path.join(base, "raw_data")  # Input folder path
    output_folder = os.path.join(base, "modified_raw_data")  # Output folder path

    # Define the required columns based on your note
    required_columns = [
        "Date", "Time", "Description", "Trigger", "Inv kW Sum (kW)", "Load kW Sum (kW)", "Solar kW (kW)", "Ambient Temp (C)",
        "Solar Radiation (W/m2)", "Inv Exp kWh", "Inv Imp kWh", "Src A Exp kWh", "Src A Imp kWh", "Src B Exp kWh",
        "Src B Imp kWh", "Site kWh (calc)", "Batt Exp kWh", "Batt Imp kWh", "Solar kWh"
    ]

    # Execute extraction
    extract_columns(input_folder, output_folder, required_columns)
