import pandas as pd
import os
import re  # Import the regex module

# Step 1: Set the path to the folder containing the data files
folder_path = os.path.join(os.getcwd(), "modified_raw_data")

# Step 2: Initialize an empty list to store DataFrames
dataframes = []

# Step 3: Load and preprocess each file
for file in os.listdir(folder_path):
    # Check if the file follows the naming convention: Arts_monthname_year
    if file.startswith("Arts_") and file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        print(f"Processing file: {file}")
        
        # Use regex to extract the month name and year
        match = re.match(r"Arts_(\w+?)(\d{4})\.csv", file)
        if match:
            month_name = match.group(1)  # Extract month name
            year = match.group(2)        # Extract year
        
            # Convert the month name to a numerical value
            month_mapping = {
                "January": 1, "February": 2, "March": 3, "April": 4,
                "May": 5, "June": 6, "July": 7, "August": 8,
                "September": 9, "October": 10, "November": 11, "December": 12
            }
            month = month_mapping.get(month_name, None)
            
            if month is None:
                print(f"Warning: Month name '{month_name}' is not recognized in file '{file}'. Skipping.")
                continue

            # Load the data into a temporary DataFrame
            temp_df = pd.read_csv(file_path)
            
            # Add 'Year' and 'Month' columns for sorting
            temp_df['Year'] = int(year)
            temp_df['Month'] = month
            
            # Handle missing values
            temp_df = temp_df.interpolate(method='linear').ffill().bfill()
            
            # Optionally, process 'calls norm DNI' if it contains categorical data
            if 'calls norm DNI' in temp_df.columns:
                temp_df['calls norm DNI'] = temp_df['calls norm DNI'].astype('category')
            
            # Append the cleaned DataFrame to the list
            dataframes.append(temp_df)
        else:
            print(f"Error: File '{file}' does not match the expected naming convention. Skipping.")

# Step 4: Combine all the DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Step 5: Sort the DataFrame chronologically by Year and Month
combined_df = combined_df.sort_values(by=['Year', 'Month'], ascending=[True, True])

# Step 6: Drop 'Year' and 'Month' columns after sorting
combined_df = combined_df.drop(columns=['Year', 'Month'])

# Step 6: Remove duplicate rows if any
combined_df = combined_df.drop_duplicates()

# Step 7: Print a summary of the cleaned and combined DataFrame
print("Combined DataFrame Shape:", combined_df.shape)
print("Combined DataFrame Columns:", combined_df.columns)
print("Combined DataFrame Preview:")
print(combined_df.head())

# Step 8: Save the final preprocessed data to a new CSV file
output_file = 'Training_Data_prev_F20_1.csv'
combined_df.to_csv(output_file, index=False)
print(f"Preprocessed data saved as '{output_file}'")
