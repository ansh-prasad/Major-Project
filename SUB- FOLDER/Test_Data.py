import pandas as pd

def extract_fault_data(input_csv, output_csv, column_name='Actual_Fault_Label', fault_label='FAULT', window=19, num_faults=2):
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Find indices where the fault occurs
    fault_indices = df[df[column_name] == fault_label].index
    
    # Limit the number of faults if specified
    if num_faults is not None:
        fault_indices = fault_indices[:num_faults]
    
    # Initialize an empty DataFrame to store the extracted rows
    extracted_rows = pd.DataFrame()
    
    # Loop through each fault occurrence
    for index in fault_indices:
        # Get the range of rows around the fault index
        start_idx = max(index - window, 0)
        end_idx = min(index + window + 1, len(df))
        
        # Append the selected rows to the extracted_rows DataFrame
        extracted_rows = pd.concat([extracted_rows, df.iloc[start_idx:end_idx]])
    
    # Drop duplicate rows in case of overlapping fault windows
    extracted_rows = extracted_rows.drop_duplicates()
    
    # Save the filtered data to a new CSV file
    extracted_rows.to_csv(output_csv, index=False)
    
    print(f"Filtered data saved to {output_csv}")

# Example usage
extract_fault_data('Test_Data.csv', 'Filtered_Test_Data.csv')