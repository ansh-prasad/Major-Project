import os
import pandas as pd

# Define the folder paths
input_folder = "test-1"
output_folder = "processed_test-1"

# Define the categories and their subfolders
categories = {
    "NORMAL DATA LOG": ["Data Log", ":"],
    "FAULT DATA LOG": [
        "Fault : Ambient Temp Sensor Fault",
        "Fault : High DC Volts",
        "Fault : Inv Sync Fault",
        "Fault : Spare Flt 6"
    ],
    "STATUS DATA LOG": [
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
        "Status : Start Source A"
    ],
    "USER DATA LOG": [
        "User : Fault Reset",
        "User : Inverter Started",
        "User : Remote GSM connected",
        "User : Remote GSM disconnected",
        "User : System Off"
    ]
}

# Helper function to sanitize folder names
def sanitize_folder_name(name):
    return name.replace(":", "_").replace("/", "_").replace("\\", "_").replace("*", "_").replace("?", "_").replace("<", "_").replace(">", "_").replace("|", "_")

# Create output folders and subfolders
for category, subfolders in categories.items():
    for subfolder in subfolders:
        sanitized_subfolder = sanitize_folder_name(subfolder)
        os.makedirs(os.path.join(output_folder, category, sanitized_subfolder), exist_ok=True)

# Process each file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):  # Ensure we process only CSV files
        file_path = os.path.join(input_folder, file_name)
        # Read the data
        df = pd.read_csv(file_path)

        # Filter and save data based on description
        for category, subfolders in categories.items():
            for subfolder in subfolders:
                # Filter rows matching the description
                filtered_data = df[df["Description"].str.contains(subfolder, na=False)]
                if not filtered_data.empty:
                    # Save the filtered data to the corresponding folder
                    sanitized_subfolder = sanitize_folder_name(subfolder)
                    output_path = os.path.join(output_folder, category, sanitized_subfolder, file_name)
                    filtered_data.to_csv(output_path, index=False)

print("Data processing complete!")
