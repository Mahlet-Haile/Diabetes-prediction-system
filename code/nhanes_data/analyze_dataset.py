"""
Script to analyze the NHANES diabetes dataset statistics
"""

import os
import pandas as pd
import numpy as np

# Directory with NHANES data files
NHANES_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the updated dataset
dataset_path = os.path.join(NHANES_DIR, "updated_nhanes_diabetes_dataset.csv")
df = pd.read_csv(dataset_path)

print("\n===== NHANES DATASET ANALYSIS =====")
print(f"Total participants: {len(df)}")

# Diabetes status counts
diabetic_count = df['diabetes'].sum()
prediabetic_count = df['prediabetes'].sum()
nondiabetic_count = len(df) - diabetic_count - prediabetic_count

print("\n----- Disease Status -----")
print(f"Diabetic:     {diabetic_count} ({diabetic_count/len(df)*100:.1f}%)")
print(f"Prediabetic:  {prediabetic_count} ({prediabetic_count/len(df)*100:.1f}%)")
print(f"Non-diabetic: {nondiabetic_count} ({nondiabetic_count/len(df)*100:.1f}%)")

# Gender distribution
# In NHANES, gender 1=Male, 2=Female
male_count = (df['gender'] == 1).sum()
female_count = (df['gender'] == 2).sum()

print("\n----- Gender Distribution -----")
print(f"Male:   {male_count} ({male_count/len(df)*100:.1f}%)")
print(f"Female: {female_count} ({female_count/len(df)*100:.1f}%)")

# Age statistics
age_min = df['age'].min()
age_max = df['age'].max()
age_mean = df['age'].mean()
age_median = df['age'].median()

print("\n----- Age Distribution -----")
print(f"Age range: {age_min} to {age_max} years")
print(f"Mean age:  {age_mean:.1f} years")
print(f"Median age: {age_median:.1f} years")

# Age distribution by decade
age_bins = [0, 18, 30, 40, 50, 60, 70, 80, 100]
age_labels = ['<18', '18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
age_dist = df['age_group'].value_counts().sort_index()

print("\n----- Age Groups -----")
for age_group, count in age_dist.items():
    print(f"{age_group}: {count} ({count/len(df)*100:.1f}%)")

# Diabetes by gender
male_diabetes = df[df['gender'] == 1]['diabetes'].sum()
female_diabetes = df[df['gender'] == 2]['diabetes'].sum()
male_prediabetes = df[df['gender'] == 1]['prediabetes'].sum()
female_prediabetes = df[df['gender'] == 2]['prediabetes'].sum()

print("\n----- Diabetes by Gender -----")
print(f"Male diabetic:    {male_diabetes} ({male_diabetes/male_count*100:.1f}% of males)")
print(f"Female diabetic:  {female_diabetes} ({female_diabetes/female_count*100:.1f}% of females)")
print(f"Male prediabetic:    {male_prediabetes} ({male_prediabetes/male_count*100:.1f}% of males)")
print(f"Female prediabetic:  {female_prediabetes} ({female_prediabetes/female_count*100:.1f}% of females)")

# Diabetes by age group
print("\n----- Diabetes by Age Group -----")
for age_group in age_dist.index:
    age_group_count = (df['age_group'] == age_group).sum()
    if age_group_count > 0:
        diabetes_in_group = df[df['age_group'] == age_group]['diabetes'].sum()
        prediabetes_in_group = df[df['age_group'] == age_group]['prediabetes'].sum()
        print(f"{age_group}: {diabetes_in_group} diabetic ({diabetes_in_group/age_group_count*100:.1f}%), "
              f"{prediabetes_in_group} prediabetic ({prediabetes_in_group/age_group_count*100:.1f}%)")

print("\n===== END OF ANALYSIS =====")
