"""
Script to remove 'ethnicity' and 'data_quality' columns from NHANES datasets
"""

import os
import pandas as pd

# Directory with NHANES data files
NHANES_DIR = os.path.dirname(os.path.abspath(__file__))

# List of datasets to modify
datasets = [
    "nhanes_diabetes_dataset.csv",
    "nhanes_training.csv",
    "nhanes_testing.csv"
]

# Columns to remove
columns_to_remove = ['ethnicity', 'data_quality']

# Process each dataset
for dataset in datasets:
    file_path = os.path.join(NHANES_DIR, dataset)
    output_path = os.path.join(NHANES_DIR, f"updated_{dataset}")
    
    if os.path.exists(file_path):
        print(f"Processing {dataset}...")
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Check which columns exist and print current columns
        existing_columns = [col for col in columns_to_remove if col in df.columns]
        print(f"  Current columns: {', '.join(df.columns.tolist())}")
        print(f"  Columns to remove: {', '.join(existing_columns)}")
        
        # Drop the specified columns if they exist
        if existing_columns:
            df = df.drop(columns=existing_columns)
            # Save to a new file
            df.to_csv(output_path, index=False)
            print(f"  Removed {len(existing_columns)} columns from {dataset}")
            print(f"  New columns: {', '.join(df.columns.tolist())}")
            print(f"  Saved to: {output_path}")
        else:
            print(f"  No columns to remove from {dataset}")
    else:
        print(f"Warning: {dataset} not found at {file_path}")

print("All datasets processed successfully.")
