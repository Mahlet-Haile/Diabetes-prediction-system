"""
Process NHANES 2017-2018 data for diabetes prediction.
This script downloads essential datasets from NHANES 2017-2018,
processes them to extract diabetes-related variables, and creates
a dataset limited to 10,000 samples for model training.
"""

import os
import pandas as pd
import numpy as np
import requests
import time

# Directory to save NHANES data
NHANES_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_SAMPLES = 10000  # Limit dataset to 10,000 samples

# Essential NHANES 2017-2018 dataset URLs - only the most relevant ones
NHANES_URLS = {
    # Demographics
    'demographics': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT',
    
    # Body Measurements
    'body_measures': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT',
    
    # Blood Pressure
    'blood_pressure': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT',
    
    # Diabetes Questionnaire
    'diabetes': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DIQ_J.XPT',
    
    # Glycohemoglobin (HbA1c)
    'hba1c': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GHB_J.XPT',
    
    # Plasma Fasting Glucose
    'glucose': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GLU_J.XPT',
    
    # Cholesterol - Total
    'cholesterol': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/TCHOL_J.XPT',
    
    # HDL Cholesterol
    'hdl': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/HDL_J.XPT',
    
    # Smoking
    'smoking': 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/SMQ_J.XPT',
}

def download_file(url, filename, max_retries=3):
    """Download a file from a URL with retry logic."""
    print(f"Downloading {filename}...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded {filename}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_retries} attempts")
                return False

def download_nhanes_data():
    """Download essential NHANES data files."""
    os.makedirs(NHANES_DIR, exist_ok=True)
    
    successful_downloads = 0
    for name, url in NHANES_URLS.items():
        filename = os.path.join(NHANES_DIR, f"{name}.XPT")
        if os.path.exists(filename):
            print(f"{filename} already exists, skipping download")
            successful_downloads += 1
        else:
            if download_file(url, filename):
                successful_downloads += 1
    
    print(f"Downloaded {successful_downloads}/{len(NHANES_URLS)} files")
    return successful_downloads == len(NHANES_URLS)

def load_xpt_file(filename):
    """Load an XPT file into a pandas DataFrame with error handling."""
    try:
        df = pd.read_sas(filename)
        print(f"Loaded {filename} with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return pd.DataFrame()

def process_nhanes_data():
    """Process NHANES data and create merged dataset limited to 10,000 samples."""
    print("Processing NHANES data...")
    
    # Load demographics data as the base
    demo_file = os.path.join(NHANES_DIR, "demographics.XPT")
    if not os.path.exists(demo_file):
        print("Demographics file not found. Please run download_nhanes_data() first.")
        return None
    
    demo_df = load_xpt_file(demo_file)
    
    # Only keep adults (18+ years)
    demo_df = demo_df[demo_df['RIDAGEYR'] >= 18].copy()
    
    # Sample up to MAX_SAMPLES if we have more
    if len(demo_df) > MAX_SAMPLES:
        demo_df = demo_df.sample(MAX_SAMPLES, random_state=42)
        print(f"Sampled {MAX_SAMPLES} participants from demographics data")
    
    # Initialize merged dataframe with essential demographic data
    merged_df = demo_df[['SEQN', 'RIDAGEYR', 'RIAGENDR']].copy()
    merged_df.rename(columns={
        'SEQN': 'participant_id',
        'RIDAGEYR': 'age',
        'RIAGENDR': 'gender',  # 1=Male, 2=Female
    }, inplace=True)
    
    # Process and merge body measurements
    bmx_file = os.path.join(NHANES_DIR, "body_measures.XPT")
    if os.path.exists(bmx_file):
        bmx_df = load_xpt_file(bmx_file)
        bmx_subset = bmx_df[['SEQN', 'BMXWT', 'BMXHT', 'BMXBMI']].copy()
        bmx_subset.rename(columns={
            'BMXWT': 'weight',  # in kg
            'BMXHT': 'height',  # in cm
            'BMXBMI': 'bmi'
        }, inplace=True)
        merged_df = pd.merge(merged_df, bmx_subset, left_on='participant_id', right_on='SEQN', how='left')
        merged_df.drop('SEQN', axis=1, inplace=True)
    
    # Process and merge blood pressure
    bpx_file = os.path.join(NHANES_DIR, "blood_pressure.XPT")
    if os.path.exists(bpx_file):
        bpx_df = load_xpt_file(bpx_file)
        # Average of first three readings for systolic and diastolic
        bpx_cols = ['SEQN']
        if 'BPXSY1' in bpx_df.columns: bpx_cols.append('BPXSY1')
        if 'BPXDI1' in bpx_df.columns: bpx_cols.append('BPXDI1')
        
        bpx_subset = bpx_df[bpx_cols].copy()
        # Rename only the columns that exist
        rename_cols = {}
        if 'BPXSY1' in bpx_subset.columns:
            rename_cols['BPXSY1'] = 'blood_pressure_systolic'
        if 'BPXDI1' in bpx_subset.columns:
            rename_cols['BPXDI1'] = 'blood_pressure_diastolic'
            
        bpx_subset.rename(columns=rename_cols, inplace=True)
        merged_df = pd.merge(merged_df, bpx_subset, left_on='participant_id', right_on='SEQN', how='left')
        merged_df.drop('SEQN', axis=1, inplace=True)
    
    # Process and merge diabetes questionnaire
    diq_file = os.path.join(NHANES_DIR, "diabetes.XPT")
    if os.path.exists(diq_file):
        diq_df = load_xpt_file(diq_file)
        diq_cols = ['SEQN']
        # Add columns only if they exist
        for col in ['DIQ010', 'DIQ160', 'DIQ170', 'DIQ180']:
            if col in diq_df.columns:
                diq_cols.append(col)
                
        diq_subset = diq_df[diq_cols].copy()
        
        # Create diabetes indicator (1=Yes, 2=No, 3=Borderline)
        if 'DIQ010' in diq_subset.columns:
            diq_subset['diabetes'] = np.where(diq_subset['DIQ010'] == 1, 1, 0)
            diq_subset['prediabetes'] = np.where(diq_subset['DIQ010'] == 3, 1, 0)
        
        # Family history of diabetes (1=Yes, 2=No)
        if 'DIQ180' in diq_subset.columns:
            diq_subset['family_history'] = np.where(diq_subset['DIQ180'] == 1, 1, 0)
        
        # Keep only the processed columns
        keep_cols = ['SEQN']
        for col in ['diabetes', 'prediabetes', 'family_history']:
            if col in diq_subset.columns:
                keep_cols.append(col)
                
        diq_final = diq_subset[keep_cols]
        merged_df = pd.merge(merged_df, diq_final, left_on='participant_id', right_on='SEQN', how='left')
        merged_df.drop('SEQN', axis=1, inplace=True)
    
    # Process and merge HbA1c
    ghb_file = os.path.join(NHANES_DIR, "hba1c.XPT")
    if os.path.exists(ghb_file):
        ghb_df = load_xpt_file(ghb_file)
        ghb_cols = ['SEQN']
        if 'LBXGH' in ghb_df.columns:
            ghb_cols.append('LBXGH')
            
        ghb_subset = ghb_df[ghb_cols].copy()
        
        # Rename only if column exists
        if 'LBXGH' in ghb_subset.columns:
            ghb_subset.rename(columns={'LBXGH': 'hba1c'}, inplace=True)
            
        merged_df = pd.merge(merged_df, ghb_subset, left_on='participant_id', right_on='SEQN', how='left')
        merged_df.drop('SEQN', axis=1, inplace=True)
    
    # Process and merge fasting glucose
    glu_file = os.path.join(NHANES_DIR, "glucose.XPT")
    if os.path.exists(glu_file):
        glu_df = load_xpt_file(glu_file)
        glu_cols = ['SEQN']
        if 'LBXGLU' in glu_df.columns:
            glu_cols.append('LBXGLU')
            
        glu_subset = glu_df[glu_cols].copy()
        
        # Rename only if column exists
        if 'LBXGLU' in glu_subset.columns:
            glu_subset.rename(columns={'LBXGLU': 'fasting_glucose'}, inplace=True)
            
        merged_df = pd.merge(merged_df, glu_subset, left_on='participant_id', right_on='SEQN', how='left')
        merged_df.drop('SEQN', axis=1, inplace=True)
    
    # Process and merge cholesterol
    chol_file = os.path.join(NHANES_DIR, "cholesterol.XPT")
    if os.path.exists(chol_file):
        chol_df = load_xpt_file(chol_file)
        chol_cols = ['SEQN']
        if 'LBXTC' in chol_df.columns:
            chol_cols.append('LBXTC')
            
        chol_subset = chol_df[chol_cols].copy()
        
        # Rename only if column exists
        if 'LBXTC' in chol_subset.columns:
            chol_subset.rename(columns={'LBXTC': 'cholesterol'}, inplace=True)
            
        merged_df = pd.merge(merged_df, chol_subset, left_on='participant_id', right_on='SEQN', how='left')
        merged_df.drop('SEQN', axis=1, inplace=True)
    
    # Process and merge HDL
    hdl_file = os.path.join(NHANES_DIR, "hdl.XPT")
    if os.path.exists(hdl_file):
        hdl_df = load_xpt_file(hdl_file)
        hdl_cols = ['SEQN']
        if 'LBDHDD' in hdl_df.columns:
            hdl_cols.append('LBDHDD')
            
        hdl_subset = hdl_df[hdl_cols].copy()
        
        # Rename only if column exists
        if 'LBDHDD' in hdl_subset.columns:
            hdl_subset.rename(columns={'LBDHDD': 'hdl'}, inplace=True)
            
        merged_df = pd.merge(merged_df, hdl_subset, left_on='participant_id', right_on='SEQN', how='left')
        merged_df.drop('SEQN', axis=1, inplace=True)
    
    # Process and merge smoking
    smq_file = os.path.join(NHANES_DIR, "smoking.XPT")
    if os.path.exists(smq_file):
        smq_df = load_xpt_file(smq_file)
        smq_cols = ['SEQN']
        for col in ['SMQ020', 'SMQ040']:
            if col in smq_df.columns:
                smq_cols.append(col)
                
        smq_subset = smq_df[smq_cols].copy()
        
        # Create smoking history categories if columns exist
        if 'SMQ020' in smq_subset.columns and 'SMQ040' in smq_subset.columns:
            # Initialize with default 'never' for everyone
            smq_subset['smoking_history'] = 'never'
            
            # Current smoker: SMQ020=1 (Yes to 100 cigarettes) and SMQ040=1 or 2 (smoke now)
            current_mask = (smq_subset['SMQ020'] == 1) & ((smq_subset['SMQ040'] == 1) | (smq_subset['SMQ040'] == 2))
            smq_subset.loc[current_mask, 'smoking_history'] = 'current'
            
            # Former smoker: SMQ020=1 (Yes to 100 cigarettes) and SMQ040=3 (not at all now)
            former_mask = (smq_subset['SMQ020'] == 1) & (smq_subset['SMQ040'] == 3)
            smq_subset.loc[former_mask, 'smoking_history'] = 'former'
            
            # Binary smoking indicator (current smoker or not)
            smq_subset['smoking'] = np.where(
                (smq_subset['SMQ020'] == 1) & ((smq_subset['SMQ040'] == 1) | (smq_subset['SMQ040'] == 2)),
                1, 0
            )
            
            # Keep only processed columns
            smq_final = smq_subset[['SEQN', 'smoking_history', 'smoking']]
            merged_df = pd.merge(merged_df, smq_final, left_on='participant_id', right_on='SEQN', how='left')
            merged_df.drop('SEQN', axis=1, inplace=True)
    
    # Fill missing values for categorical variables
    if 'smoking_history' in merged_df.columns:
        merged_df['smoking_history'].fillna('no_info', inplace=True)
    
    # Convert gender to match your system (1=Male, 2=Female)
    if 'gender' in merged_df.columns:
        # Your code might expect a string value like 'M' or 'F'
        # merged_df['gender'] = merged_df['gender'].map({1: 'M', 2: 'F'})
        # Or keep as 1, 2 if that's what your system expects
        pass
    
    # Ensure we have diabetes status
    if 'diabetes' in merged_df.columns:
        # Drop rows with missing diabetes status (our target variable)
        merged_df = merged_df.dropna(subset=['diabetes'])
    else:
        print("Warning: No diabetes status column found in the processed data!")
    
    # Save the processed data
    output_file = os.path.join(NHANES_DIR, "nhanes_diabetes_dataset.csv")
    merged_df.to_csv(output_file, index=False)
    
    # Split into training and testing sets (80/20 split)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42, stratify=merged_df['diabetes'] if 'diabetes' in merged_df.columns else None)
    
    train_file = os.path.join(NHANES_DIR, "nhanes_training.csv")
    test_file = os.path.join(NHANES_DIR, "nhanes_testing.csv")
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    # Print summary statistics
    print(f"\nProcessed NHANES data saved to {output_file}")
    print(f"Total samples: {len(merged_df)}")
    print(f"Variables: {', '.join(merged_df.columns.tolist())}")
    print(f"Training data saved to {train_file} ({len(train_df)} samples)")
    print(f"Testing data saved to {test_file} ({len(test_df)} samples)")
    
    if 'diabetes' in merged_df.columns:
        diabetes_count = merged_df['diabetes'].sum()
        print(f"Diabetes cases: {diabetes_count} ({diabetes_count/len(merged_df)*100:.1f}%)")
    
    return merged_df

def main():
    print("Starting NHANES 2017-2018 data download and processing...")
    download_success = download_nhanes_data()
    
    if download_success:
        print("\nAll files downloaded successfully. Processing data...")
        process_nhanes_data()
        print("\nNHANES data processing complete.")
    else:
        print("\nSome files could not be downloaded. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
