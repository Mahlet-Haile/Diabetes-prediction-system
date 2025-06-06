"""
Create a balanced NHANES dataset with realistic disease prevalence
using stratified sampling to maintain proper demographic distributions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Directory with NHANES data files
NHANES_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the updated dataset
dataset_path = os.path.join(NHANES_DIR, "updated_nhanes_diabetes_dataset.csv")
print(f"Loading dataset from {dataset_path}")
df = pd.read_csv(dataset_path)

print("\n===== CREATING BALANCED NHANES DATASET =====")
print(f"Original dataset: {len(df)} participants")

# First, let's analyze the original dataset
diabetic = df[df['diabetes'] == 1]
prediabetic = df[(df['diabetes'] == 0) & (df['prediabetes'] == 1)]
nondiabetic = df[(df['diabetes'] == 0) & (df['prediabetes'] == 0)]

print(f"\nOriginal distribution:")
print(f"Diabetic:     {len(diabetic)} ({len(diabetic)/len(df)*100:.1f}%)")
print(f"Prediabetic:  {len(prediabetic)} ({len(prediabetic)/len(df)*100:.1f}%)")
print(f"Non-diabetic: {len(nondiabetic)} ({len(nondiabetic)/len(df)*100:.1f}%)")

# Target number of prediabetic samples (around 38% of total)
target_prediabetic = 1100

# Prepare stratification for balanced sampling of prediabetic cases
# We'll create age groups and combine with gender for stratification
prediabetic['age_group'] = pd.cut(prediabetic['age'], 
                                  bins=[0, 30, 40, 50, 60, 70, 80, 100], 
                                  labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'])
prediabetic['strata'] = prediabetic['age_group'].astype(str) + '_' + prediabetic['gender'].astype(str)

# Print distribution before sampling
print("\nPrediabetic cases distribution before sampling:")
strata_counts = prediabetic['strata'].value_counts().sort_index()
for strata, count in strata_counts.items():
    print(f"  {strata}: {count} ({count/len(prediabetic)*100:.1f}%)")

# Calculate target counts for each stratum to maintain proportions
strata_proportions = prediabetic['strata'].value_counts(normalize=True).sort_index()
target_counts = (strata_proportions * target_prediabetic).round().astype(int)

# Adjust if the sum doesn't match the target due to rounding
diff = target_prediabetic - target_counts.sum()
if diff != 0:
    # Add/subtract the difference from the largest strata
    largest_strata = strata_counts.idxmax()
    target_counts[largest_strata] += diff

# Perform stratified sampling
sampled_prediabetic = pd.DataFrame()
for strata, target in target_counts.items():
    strata_df = prediabetic[prediabetic['strata'] == strata]
    if len(strata_df) <= target:
        # If we have fewer or equal samples than target, keep all of them
        sampled_from_strata = strata_df
    else:
        # Otherwise, sample the target number
        sampled_from_strata = strata_df.sample(n=target, random_state=42)
    sampled_prediabetic = pd.concat([sampled_prediabetic, sampled_from_strata])

# Drop the temporary columns used for stratification
sampled_prediabetic = sampled_prediabetic.drop(columns=['age_group', 'strata'])

# Print distribution after sampling
print(f"\nPrediabetic cases after stratified sampling: {len(sampled_prediabetic)}")
print(f"Target was: {target_prediabetic}")

# Combine all parts to create the balanced dataset
balanced_df = pd.concat([diabetic, sampled_prediabetic, nondiabetic])

# Final statistics
print(f"\nFinal balanced dataset: {len(balanced_df)} participants")
print(f"Diabetic:     {len(diabetic)} ({len(diabetic)/len(balanced_df)*100:.1f}%)")
print(f"Prediabetic:  {len(sampled_prediabetic)} ({len(sampled_prediabetic)/len(balanced_df)*100:.1f}%)")
print(f"Non-diabetic: {len(nondiabetic)} ({len(nondiabetic)/len(balanced_df)*100:.1f}%)")

# Verify gender balance
male_count = (balanced_df['gender'] == 1).sum()
female_count = (balanced_df['gender'] == 2).sum()
print(f"\nGender distribution in balanced dataset:")
print(f"Male:   {male_count} ({male_count/len(balanced_df)*100:.1f}%)")
print(f"Female: {female_count} ({female_count/len(balanced_df)*100:.1f}%)")

# Verify age distribution
age_bins = [0, 18, 30, 40, 50, 60, 70, 80, 100]
age_labels = ['<18', '18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
balanced_df['age_group'] = pd.cut(balanced_df['age'], bins=age_bins, labels=age_labels, right=False)
age_dist = balanced_df['age_group'].value_counts().sort_index()

print("\nAge distribution in balanced dataset:")
for age_group, count in age_dist.items():
    print(f"{age_group}: {count} ({count/len(balanced_df)*100:.1f}%)")

# Save the balanced dataset
balanced_output_path = os.path.join(NHANES_DIR, "balanced_nhanes_diabetes_dataset.csv")
balanced_df.drop(columns=['age_group'], inplace=True)  # Remove temporary column
balanced_df.to_csv(balanced_output_path, index=False)

# Split into training and testing sets (80/20 split with stratification)
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(
    balanced_df, 
    test_size=0.2, 
    stratify=balanced_df[['diabetes', 'prediabetes']], 
    random_state=42
)

train_output_path = os.path.join(NHANES_DIR, "balanced_nhanes_training.csv")
test_output_path = os.path.join(NHANES_DIR, "balanced_nhanes_testing.csv")

train_df.to_csv(train_output_path, index=False)
test_df.to_csv(test_output_path, index=False)

print(f"\nBalanced dataset saved to {balanced_output_path}")
print(f"Training data saved to {train_output_path} ({len(train_df)} participants)")
print(f"Testing data saved to {test_output_path} ({len(test_df)} participants)")
print("\n===== DONE =====")
