import pandas as pd
import os

# Load the CSV file
file_path = 'Combined_full_data.csv'
data = pd.read_csv(file_path)

# Ensure the 'Description' column exists
if 'Description' not in data.columns:
    print("The 'Description' column is not present in the dataset.")
else:
    # Define the main folder and specific events
    main_folder = "test"
    event_mapping = {
        "Status": [
            "Status : Parallel Mode",
            "Status : Source A Stopped",
            "Status : Source A Warming Up",
            "Status : Solar Control Disabled",
            "Status : Auto Inv Only Mode",
            "Status : Inverter Masked",
            "Status : Source A CB Closed",
            "Status : Source A CB Opened",
            "Status : Inverter Started",
            "Status : Source A Cooling Down",
            "Status : Solar Control Enabled",
            "Status : Inverter CB Opened",
            "Status : Inverter Stopped",
            "Status : Inverter CB Closed",
            "Status : Battery Charge Stage 4",
            "Status : Start Source A",
            "Status : Low Battery Bulk Enable",
            "Status : Battery Charge Stage 1",
            "Status : Battery Charge Stage 2",
            "Status : Battery Charge Stage 3",
            ],
        "User": [
            "User : System Off",
            "User : Fault Reset",
            "User : Remote GSM connected",
            "User : Remote GSM disconnected",
            "User : Inverter Started",
        ],
        "Fault": [
            "Fault : Inv Sync Fault",
            "Fault : Ambient Temp Sensor Fault",
            "Fault : High DC Volts",
            "Fault : Spare Flt 6",
        ]
    }

    # Create the main folder
    os.makedirs(main_folder, exist_ok=True)

    # Iterate through event types (Status, User, Fault)
    for folder_name, events in event_mapping.items():
        # Create a subfolder for the event type
        subfolder_path = os.path.join(main_folder, folder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # Iterate through each specific event in the category
        for event in events:
            # Find the index of rows where 'Description' matches the specific event
            event_indices = data[data['Description'].str.contains(event, na=False, case=False)].index.tolist()

            # Prepare the list to store the data with surrounding rows
            surrounding_data = []
 
            for index in event_indices:
                # Get 5 rows before and 5 rows after the event entry
                start_idx = max(index - 20, 0)  # Ensure the index doesn't go negative
                end_idx = min(index + 21, len(data))  

                # Extract the relevant rows
                surrounding_rows = data.iloc[start_idx:end_idx]
                surrounding_data.append(surrounding_rows)

                # Add a gap of 5 empty rows (NaN values)
                gap = pd.DataFrame([[''] * len(data.columns)] * 5, columns=data.columns)
                surrounding_data.append(gap)

            # Concatenate all the data into a single DataFrame
            result_data = pd.concat(surrounding_data, ignore_index=True)

            # Save filtered data to a CSV file in the corresponding subfolder
            output_file = os.path.join(subfolder_path, f"{event.replace(' : ', '_').replace(' ', '_')}.csv")
            result_data.to_csv(output_file, index=False)
            print(f"Saved rows containing '{event}' and surrounding rows with gap to {output_file}")