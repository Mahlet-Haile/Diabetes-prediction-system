"""
Create an epidemiologically accurate NHANES dataset with realistic disease prevalence:
- 12.1% diabetic
- 35.0% prediabetic
- 52.9% non-diabetic

Using stratified sampling to maintain realistic demographic distributions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Directory with NHANES data files
NHANES_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the original full dataset
dataset_path = os.path.join(NHANES_DIR, "nhanes_diabetes_dataset.csv")
print(f"Loading dataset from {dataset_path}")
df = pd.read_csv(dataset_path)

print("\n===== CREATING EPIDEMIOLOGICALLY ACCURATE NHANES DATASET =====")
print(f"Original dataset: {len(df)} participants")

# First, let's analyze the original dataset
diabetic = df[df['diabetes'] == 1]
prediabetic = df[(df['diabetes'] == 0) & (df['prediabetes'] == 1)]
nondiabetic = df[(df['diabetes'] == 0) & (df['prediabetes'] == 0)]

print(f"\nOriginal distribution:")
print(f"Diabetic:     {len(diabetic)} ({len(diabetic)/len(df)*100:.1f}%)")
print(f"Prediabetic:  {len(prediabetic)} ({len(prediabetic)/len(df)*100:.1f}%)")
print(f"Non-diabetic: {len(nondiabetic)} ({len(nondiabetic)/len(df)*100:.1f}%)")

# Target number of prediabetic samples (35% of total)
target_prediabetic = 641

# Filter prediabetic cases to include only those up to age 85
prediabetic = prediabetic[prediabetic['age'] <= 85]
print(f"\nPrediabetic cases after age filter (<=85 years): {len(prediabetic)}")

# Prepare stratification for balanced sampling of prediabetic cases
# We'll create age groups and combine with gender for stratification
age_bins = [0, 30, 40, 50, 60, 70, 80, 85]
age_labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-85']
prediabetic['age_group'] = pd.cut(prediabetic['age'], bins=age_bins, labels=age_labels)
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

# Check gender balance in target counts
male_targets = sum([count for strata, count in target_counts.items() if strata.endswith('_1')])
female_targets = sum([count for strata, count in target_counts.items() if strata.endswith('_2')])
print(f"\nTarget gender distribution in prediabetic sample:")
print(f"  Male: {male_targets} ({male_targets/target_prediabetic*100:.1f}%)")
print(f"  Female: {female_targets} ({female_targets/target_prediabetic*100:.1f}%)")

# Adjust to make gender distribution closer to 50/50 if needed
if abs(male_targets - female_targets) > 20:  # If difference is significant
    print("  Adjusting gender balance to be closer to 50/50...")
    if male_targets > female_targets:
        # Need to increase female counts and decrease male counts
        gender_diff = (male_targets - female_targets) // 2
        # Find largest male and female strata
        male_strata = [s for s in target_counts.index if s.endswith('_1')]
        female_strata = [s for s in target_counts.index if s.endswith('_2')]
        male_strata.sort(key=lambda s: target_counts[s], reverse=True)
        female_strata.sort(key=lambda s: target_counts[s], reverse=True)
        
        # Adjust counts
        for i in range(min(gender_diff, len(male_strata), len(female_strata))):
            target_counts[male_strata[i]] -= 1
            target_counts[female_strata[i]] += 1
    else:
        # Need to increase male counts and decrease female counts
        gender_diff = (female_targets - male_targets) // 2
        # Find largest male and female strata
        male_strata = [s for s in target_counts.index if s.endswith('_1')]
        female_strata = [s for s in target_counts.index if s.endswith('_2')]
        male_strata.sort(key=lambda s: target_counts[s], reverse=True)
        female_strata.sort(key=lambda s: target_counts[s], reverse=True)
        
        # Adjust counts
        for i in range(min(gender_diff, len(male_strata), len(female_strata))):
            target_counts[male_strata[i]] += 1
            target_counts[female_strata[i]] -= 1
    
    # Recalculate gender distribution
    male_targets = sum([count for strata, count in target_counts.items() if strata.endswith('_1')])
    female_targets = sum([count for strata, count in target_counts.items() if strata.endswith('_2')])
    print(f"  Adjusted gender distribution:")
    print(f"    Male: {male_targets} ({male_targets/target_prediabetic*100:.1f}%)")
    print(f"    Female: {female_targets} ({female_targets/target_prediabetic*100:.1f}%)")

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
final_df = pd.concat([diabetic, sampled_prediabetic, nondiabetic])

# Final statistics
print(f"\nFinal realistic dataset: {len(final_df)} participants")
print(f"Diabetic:     {len(diabetic)} ({len(diabetic)/len(final_df)*100:.1f}%)")
print(f"Prediabetic:  {len(sampled_prediabetic)} ({len(sampled_prediabetic)/len(final_df)*100:.1f}%)")
print(f"Non-diabetic: {len(nondiabetic)} ({len(nondiabetic)/len(final_df)*100:.1f}%)")

# Verify gender balance
male_count = (final_df['gender'] == 1).sum()
female_count = (final_df['gender'] == 2).sum()
print(f"\nGender distribution in final dataset:")
print(f"Male:   {male_count} ({male_count/len(final_df)*100:.1f}%)")
print(f"Female: {female_count} ({female_count/len(final_df)*100:.1f}%)")

# Verify age distribution
age_bins = [0, 18, 30, 40, 50, 60, 70, 80, 85, 100]
age_labels = ['<18', '18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-85', '85+']
final_df['age_group'] = pd.cut(final_df['age'], bins=age_bins, labels=age_labels)
age_dist = final_df['age_group'].value_counts().sort_index()

print("\nAge distribution in final dataset:")
for age_group, count in age_dist.items():
    print(f"{age_group}: {count} ({count/len(final_df)*100:.1f}%)")

# Check diabetes by age group
print("\nDiabetes prevalence by age group:")
for age_group in age_dist.index:
    age_group_count = (final_df['age_group'] == age_group).sum()
    if age_group_count > 0:
        diabetes_in_group = final_df[final_df['age_group'] == age_group]['diabetes'].sum()
        prediabetes_in_group = final_df[(final_df['age_group'] == age_group) & 
                                        (final_df['diabetes'] == 0) & 
                                        (final_df['prediabetes'] == 1)].shape[0]
        print(f"{age_group}: {diabetes_in_group} diabetic ({diabetes_in_group/age_group_count*100:.1f}%), "
              f"{prediabetes_in_group} prediabetic ({prediabetes_in_group/age_group_count*100:.1f}%)")

# Save the final realistic dataset
output_path = os.path.join(NHANES_DIR, "realistic_nhanes_diabetes_dataset.csv")
final_df = final_df.drop(columns=['age_group'])  # Remove temporary column
final_df.to_csv(output_path, index=False)

# Split into training and testing sets (80/20 split with stratification)
train_df, test_df = train_test_split(
    final_df, 
    test_size=0.2, 
    stratify=final_df[['diabetes', 'prediabetes']], 
    random_state=42
)

train_output_path = os.path.join(NHANES_DIR, "realistic_nhanes_training.csv")
test_output_path = os.path.join(NHANES_DIR, "realistic_nhanes_testing.csv")

train_df.to_csv(train_output_path, index=False)
test_df.to_csv(test_output_path, index=False)

print(f"\nRealistic dataset saved to {output_path}")
print(f"Training data saved to {train_output_path} ({len(train_df)} participants)")
print(f"Testing data saved to {test_output_path} ({len(test_df)} participants)")
print("\n===== DONE =====")
