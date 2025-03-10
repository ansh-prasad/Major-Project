import pandas as pd
import numpy as np

# Original dataset (manually extracted from the text)
data = [
    ["23/05/22", "08:20:02.55", "Status : Source A Warming Up", "SrcA Startup", 0.7, 0.8, 0, 31, 34, 185095, 56080.2, 44372.3, 283438, 0, 0, 368257.3, 16360.4, 75694.5, 208114.3, "WARNING"],
    ["23/05/22", "08:20:41.37", "Status : Source A Warming Up", "SrcA Startup", 0.8, 0.8, 0, 30.9, 37, 185095, 56080.2, 44372.3, 283438, 0, 0, 368257.3, 16360.4, 75694.5, 208114.3, "WARNING"],
    ["23/05/22", "08:20:52.36", "Fault : Inv Sync Fault", "Inv Request Source Tracking", 2, 1.9, 0, 30.9, 38, 185095, 56080.2, 44372.3, 283438, 0, 0, 368257.3, 16360.4, 75694.5, 208114.3, "FAULT"],
    ["23/05/22", "08:21:42.96", "Status : Source A Warming Up", "SrcA Startup", 1.9, 1.9, 0.1, 31, 46, 185095, 56080.2, 44372.3, 283438, 0, 0, 368257.3, 16360.5, 75694.5, 208114.3, "WARNING"],
    ["23/05/22", "08:22:01.36", "Status : Parallel Mode", "Operation Auto Inverter Only", 0.2, 0.8, 0.2, 31.2, 48, 185095, 56080.2, 44372.3, 283438, 0, 0, 368257.3, 16360.5, 75694.5, 208114.3, "WARNING"],
    ["23/05/22", "08:22:40.68", "Status : Source A Warming Up", "SrcA Startup", 0.9, 0.8, 0.3, 30.8, 56, 185095, 56080.2, 44372.3, 283438, 0, 0, 368257.3, 16360.5, 75694.5, 208114.3, "WARNING"],
    ["23/05/22", "08:23:19.36", "Status : Source A Warming Up", "SrcA Startup", 1, 1, 0.5, 30.7, 65, 185095, 56080.2, 44372.3, 283438, 0, 0, 368257.3, 16360.5, 75694.5, 208114.3, "WARNING"],
    ["23/05/22", "08:23:33.96", "Fault : Inv Sync Fault", "Inv Request Source Tracking", 2.2, 2.2, 0.6, 30.5, 69, 185095, 56080.2, 44372.3, 283438, 0, 0, 368257.3, 16360.5, 75694.5, 208114.3, "FAULT"],
    ["23/05/22", "08:23:52.00", "Data Log", "Log Period", 1.3, 2.1, 0.7, 30.7, 73, 185095, 56080.2, 44372.3, 283438, 0, 0, 368257.3, 16360.6, 75694.5, 208114.3, "NORMAL"],
    ["23/05/22", "08:26:25.45", "Status : Parallel Mode", "Operation Auto Inverter Only", -0.1, 2.3, 1.3, 29.6, 105, 185095, 56080.2, 44372.3, 283438.1, 0, 0, 368257.4, 16360.6, 75694.6, 208114.4, "WARNING"],
]

# Convert data into a DataFrame
columns = ["Date", "Time", "Status", "Operation", "Val1", "Val2", "Val3", "Temp", "Load", "Metric1", "Metric2", "Metric3",
           "Metric4", "Metric5", "Metric6", "Metric7", "Metric8", "Metric9", "Metric10", "Level"]

df = pd.DataFrame(data, columns=columns)

# Convert Time to datetime for interpolation
df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%y %H:%M:%S.%f")

# Create new interpolated timestamps
new_rows = []
for i in range(len(df) - 1):
    start_row = df.iloc[i]
    end_row = df.iloc[i + 1]
    
    for j in range(1, 5):  # Generate 4 new points between each pair
        factor = j / 5
        new_time = start_row["Timestamp"] + factor * (end_row["Timestamp"] - start_row["Timestamp"])
        new_values = start_row[4:-1] + factor * (end_row[4:-1] - start_row[4:-1])  # Linear interpolation
        
        new_row = start_row.copy()
        new_row["Timestamp"] = new_time
        new_row.iloc[4:-1] = new_values  # Replace numerical values
        new_rows.append(new_row)

# Append new rows to the original dataset
df = pd.concat([df] + new_rows, ignore_index=True).sort_values("Timestamp")

# Convert back to string format for readability
df["Date"] = df["Timestamp"].dt.strftime("%d/%m/%y")
df["Time"] = df["Timestamp"].dt.strftime("%H:%M:%S.%f").str[:-3]  # Keep milliseconds

# Drop the Timestamp column
df = df.drop(columns=["Timestamp"])

df.head(20)  # Display first 20 rows of the updated dataset
