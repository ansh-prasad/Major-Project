import os
import pandas as pd

def get_month_number(filename):
    """Extract month and year from filename and return a sortable key."""
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }

    try:
        parts = filename.replace(".csv", "").split("_")
        if len(parts) < 3:
            return None  # Skip invalid files

        month_name, year = parts[1], parts[2]

        if month_name not in month_map or not year.isdigit():
            return None  # Skip invalid files

        return (int(year), month_map[month_name])
    
    except:
        return None  # Skip any unexpected errors

def merge_sorted_csv(input_folder, output_file):
    """Merge all CSV files in sorted month-year order."""
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]

    # Filter and sort valid files
    valid_files = sorted((file for file in csv_files if get_month_number(file)), key=get_month_number)

    if not valid_files:
        print("No valid CSV files found. Exiting without merging.")
        return
    
    # Read and merge all valid CSV files
    dataframes = [pd.read_csv(os.path.join(input_folder, file)) for file in valid_files]
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save merged CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(valid_files)} CSV files into {output_file} in correct order.")

# Specify the folder containing CSV files and the output file
input_folder = "Modified_Data"
output_file = "Merged_data.csv"

merge_sorted_csv(input_folder, output_file)
