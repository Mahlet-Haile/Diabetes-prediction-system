"""
Remove ethnicity and data_quality columns from the realistic NHANES dataset files.
"""

import os
import pandas as pd

# Directory with NHANES data files
NHANES_DIR = os.path.dirname(os.path.abspath(__file__))

# List of files to process
dataset_files = [
    "realistic_nhanes_diabetes_dataset.csv",
    "realistic_nhanes_training.csv",
    "realistic_nhanes_testing.csv"
]

# Columns to remove
columns_to_remove = ['ethnicity', 'data_quality']

# Process each file
for filename in dataset_files:
    file_path = os.path.join(NHANES_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    print(f"Processing {filename}...")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Check if columns exist in the dataset
    columns_removed = []
    for column in columns_to_remove:
        if column in df.columns:
            df = df.drop(column, axis=1)
            columns_removed.append(column)
    
    # Save the updated file
    df.to_csv(file_path, index=False)
    
    # Report results
    if columns_removed:
        print(f"  Removed columns: {', '.join(columns_removed)}")
    else:
        print(f"  No columns were removed (columns not found in dataset)")
    print(f"  Updated file saved with {len(df)} rows and {len(df.columns)} columns")

print("\nAll files processed successfully.")
