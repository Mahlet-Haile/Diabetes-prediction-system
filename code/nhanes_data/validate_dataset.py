"""
Validate the realistic NHANES diabetes dataset by examining:
1. Value distributions of key clinical variables
2. Label relationships and mutual exclusivity
3. Class balance
4. Data quality indicators
5. Encoding of categorical variables
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Directory with NHANES data files
NHANES_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the realistic dataset
dataset_path = os.path.join(NHANES_DIR, "realistic_nhanes_diabetes_dataset.csv")
print(f"Loading dataset from {dataset_path}")
df = pd.read_csv(dataset_path)

print("\n===== NHANES DATASET VALIDATION =====")
print(f"Dataset shape: {df.shape}")

# 1. Value Distributions for Key Clinical Variables
print("\n===== 1. VALUE DISTRIBUTIONS =====")
clinical_vars = ['hba1c', 'fasting_glucose', 'bmi', 'blood_pressure_systolic', 
                'blood_pressure_diastolic', 'total_cholesterol', 'hdl', 'ldl', 'triglycerides']

for var in clinical_vars:
    if var in df.columns:
        stats_data = df[var].describe()
        print(f"\n{var.upper()} Statistics:")
        print(f"  Count: {stats_data['count']}")
        print(f"  Mean: {stats_data['mean']:.2f}")
        print(f"  Std Dev: {stats_data['std']:.2f}")
        print(f"  Min: {stats_data['min']:.2f}")
        print(f"  25%: {stats_data['25%']:.2f}")
        print(f"  50% (Median): {stats_data['50%']:.2f}")
        print(f"  75%: {stats_data['75%']:.2f}")
        print(f"  Max: {stats_data['max']:.2f}")
        
        # Check for outliers (values outside 3 standard deviations)
        mean = stats_data['mean']
        std = stats_data['std']
        outliers = df[(df[var] < mean - 3*std) | (df[var] > mean + 3*std)][var]
        print(f"  Outliers (outside 3 std devs): {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
        
        # Biological plausibility check
        plausible_ranges = {
            'hba1c': (3.5, 15.0),            # % (hemoglobin A1c)
            'fasting_glucose': (50, 500),     # mg/dL
            'bmi': (13, 70),                  # kg/m²
            'blood_pressure_systolic': (70, 220),  # mmHg
            'blood_pressure_diastolic': (40, 130), # mmHg
            'total_cholesterol': (80, 400),   # mg/dL
            'hdl': (20, 120),                 # mg/dL
            'ldl': (30, 300),                 # mg/dL
            'triglycerides': (30, 1000)       # mg/dL
        }
        
        if var in plausible_ranges:
            min_val, max_val = plausible_ranges[var]
            implausible = df[(df[var] < min_val) | (df[var] > max_val)][var]
            print(f"  Biologically implausible values (outside {min_val}-{max_val}): {len(implausible)} ({len(implausible)/len(df)*100:.2f}%)")

# 2. Label Relationships
print("\n===== 2. LABEL RELATIONSHIPS =====")
# Create a crosstab of diabetes and prediabetes
label_cross = pd.crosstab(df['diabetes'], df['prediabetes'], 
                         rownames=['Diabetes'], colnames=['Prediabetes'])
print("Diabetes vs Prediabetes Crosstab:")
print(label_cross)

# Check for mutual exclusivity
both_conditions = df[(df['diabetes'] == 1) & (df['prediabetes'] == 1)]
print(f"\nParticipants with both diabetes and prediabetes: {len(both_conditions)}")
if len(both_conditions) > 0:
    print("WARNING: Labels are not mutually exclusive. Some participants are marked as having both conditions.")
else:
    print("Labels are mutually exclusive. No participants have both conditions.")

# 3. Class Balance
print("\n===== 3. CLASS BALANCE =====")
# Count for each class
diabetic = df[df['diabetes'] == 1]
prediabetic = df[(df['diabetes'] == 0) & (df['prediabetes'] == 1)]
nondiabetic = df[(df['diabetes'] == 0) & (df['prediabetes'] == 0)]

print(f"Diabetic:     {len(diabetic)} ({len(diabetic)/len(df)*100:.1f}%)")
print(f"Prediabetic:  {len(prediabetic)} ({len(prediabetic)/len(df)*100:.1f}%)")
print(f"Non-diabetic: {len(nondiabetic)} ({len(nondiabetic)/len(df)*100:.1f}%)")

# Check if class balance meets target proportions
target_diabetic = 0.121  # 12.1%
target_prediabetic = 0.35  # 35.0%
target_nondiabetic = 0.529  # 52.9%

actual_diabetic = len(diabetic)/len(df)
actual_prediabetic = len(prediabetic)/len(df)
actual_nondiabetic = len(nondiabetic)/len(df)

print("\nComparison to Target Proportions:")
print(f"Diabetic:     Target: {target_diabetic*100:.1f}%, Actual: {actual_diabetic*100:.1f}%, Diff: {(actual_diabetic-target_diabetic)*100:.2f}%")
print(f"Prediabetic:  Target: {target_prediabetic*100:.1f}%, Actual: {actual_prediabetic*100:.1f}%, Diff: {(actual_prediabetic-target_prediabetic)*100:.2f}%")
print(f"Non-diabetic: Target: {target_nondiabetic*100:.1f}%, Actual: {actual_nondiabetic*100:.1f}%, Diff: {(actual_nondiabetic-target_nondiabetic)*100:.2f}%")

# 4. Data Quality Indicators
print("\n===== 4. DATA QUALITY INDICATORS =====")
# Check if data_quality column exists
if 'data_quality' in df.columns:
    quality_counts = df['data_quality'].value_counts()
    print("Data Quality Distribution:")
    for quality, count in quality_counts.items():
        print(f"  {quality}: {count} ({count/len(df)*100:.1f}%)")
else:
    print("No 'data_quality' column found in the dataset.")
    
# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values by Column:")
for col, missing in missing_values.items():
    if missing > 0:
        print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")
if missing_values.sum() == 0:
    print("  No missing values found in the dataset.")

# 5. Encoding of Categorical Variables
print("\n===== 5. CATEGORICAL VARIABLE ENCODING =====")
categorical_vars = ['gender', 'ethnicity', 'smoking', 'alcohol_use', 
                    'hypertension', 'family_history', 'fatigue', 
                    'polyuria', 'polydipsia']

for var in categorical_vars:
    if var in df.columns:
        value_counts = df[var].value_counts().sort_index()
        print(f"\n{var.upper()} Encoding:")
        print(f"  Unique values: {sorted(df[var].unique())}")
        print(f"  Value distribution:")
        for val, count in value_counts.items():
            print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")
        
        # Provide interpretation for known encodings
        if var == 'gender':
            print("  Interpretation: 1=Male, 2=Female (NHANES encoding)")
        elif var == 'ethnicity' and 'ethnicity' in df.columns:
            print("  Interpretation: 1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White, "
                  "4=Non-Hispanic Black, 6=Non-Hispanic Asian, 7=Other/Multi (NHANES encoding)")
        elif var in ['smoking', 'alcohol_use', 'hypertension', 'family_history', 
                     'fatigue', 'polyuria', 'polydipsia']:
            print("  Interpretation: 0=No, 1=Yes (Binary encoding)")

# Check correlations between features and target variables
print("\n===== 6. FEATURE CORRELATIONS =====")
# Select numeric columns for correlation
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
# Calculate correlation with diabetes and prediabetes
corr_with_diabetes = df[numeric_cols].corr()['diabetes'].sort_values(ascending=False)
print("\nTop correlations with diabetes:")
print(corr_with_diabetes.head(10))

if 'prediabetes' in numeric_cols:
    corr_with_prediabetes = df[numeric_cols].corr()['prediabetes'].sort_values(ascending=False)
    print("\nTop correlations with prediabetes:")
    print(corr_with_prediabetes.head(10))

# Clinical validity check: HbA1c vs Fasting Glucose correlation
if 'hba1c' in df.columns and 'fasting_glucose' in df.columns:
    hba1c_glucose_corr = df['hba1c'].corr(df['fasting_glucose'])
    print(f"\nCorrelation between HbA1c and Fasting Glucose: {hba1c_glucose_corr:.3f}")
    if hba1c_glucose_corr > 0.6:
        print("  Strong positive correlation as expected clinically.")
    else:
        print("  WARNING: Correlation lower than clinically expected.")

print("\n===== VALIDATION COMPLETE =====")
