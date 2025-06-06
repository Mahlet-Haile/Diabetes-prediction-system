"""
Simplified training script for NHANES diabetes models.
This standalone script trains models on the realistic NHANES dataset
and saves them for use in the Diabetes Prediction System.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Base paths
base_dir = os.path.dirname(os.path.abspath(__file__))
saved_models_dir = os.path.join(base_dir, 'saved_models')
os.makedirs(saved_models_dir, exist_ok=True)

# Model filenames
diabetes_model_path = os.path.join(saved_models_dir, 'diabetes_model_nhanes.joblib')
diabetes_type_model_path = os.path.join(saved_models_dir, 'diabetes_type_model_nhanes.joblib')

def train_simplified_models():
    """Simplified model training function that focuses on reliability over complexity"""
    print("Starting simplified NHANES model training...")
    
    # Find the realistic NHANES dataset
    potential_dirs = [
        os.path.join(os.path.dirname(base_dir), 'nhanes_data'),
        os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'nhanes_data'),
        r'C:\Users\hi\Desktop\GDPS\nhanes_data',
        r'C:\Users\hi\Desktop\GDPS\code\nhanes_data'
    ]
    
    print("Searching for dataset in directories:")
    dataset_path = None
    for dir_path in potential_dirs:
        print(f"  - {dir_path}")
        potential_path = os.path.join(dir_path, 'realistic_nhanes_diabetes_dataset.csv')
        if os.path.exists(potential_path):
            dataset_path = potential_path
            print(f"Found dataset at: {dataset_path}")
            break
    
    if not dataset_path:
        print("ERROR: Could not find the realistic NHANES dataset")
        return False
    
    # Load the dataset
    print(f"Loading dataset from {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded successfully with {len(df)} rows")
    except Exception as e:
        print(f"ERROR loading dataset: {str(e)}")
        return False
    
    # Print basic info
    print("\nDataset overview:")
    print(f"  - Columns: {df.columns.tolist()}")
    print(f"  - Diabetic cases: {df['diabetes'].sum()} ({df['diabetes'].sum()/len(df)*100:.1f}%)")
    if 'prediabetes' in df.columns:
        print(f"  - Prediabetic cases: {df['prediabetes'].sum()} ({df['prediabetes'].sum()/len(df)*100:.1f}%)")
    
    # Create minimal feature set that should be available in any dataset
    safe_features = ['age', 'gender', 'bmi', 'hba1c', 'fasting_glucose']
    
    # Add any additional features that exist
    extra_features = ['blood_pressure_systolic', 'blood_pressure_diastolic', 
                     'hdl', 'ldl', 'triglycerides', 'total_cholesterol',
                     'smoking', 'alcohol_use', 'hypertension', 'family_history',
                     'fatigue', 'polyuria', 'polydipsia']
    
    # Filter to only include columns that exist
    all_features = safe_features + [f for f in extra_features if f in df.columns]
    print(f"\nUsing features: {all_features}")
    
    # Split data for diabetes model
    X = df[all_features]
    y_diabetes = df['diabetes']
    
    # Simple preprocessing - fill missing values with mean/mode
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])
    
    # Train diabetes model with simple RandomForest
    print("\nTraining diabetes model...")
    diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
    diabetes_model.fit(X, y_diabetes)
    
    # Evaluate on the same data (just for verification)
    y_pred = diabetes_model.predict(X)
    accuracy = accuracy_score(y_diabetes, y_pred)
    print(f"Diabetes model accuracy: {accuracy:.4f}")
    
    # Save the model
    joblib.dump(diabetes_model, diabetes_model_path)
    print(f"Diabetes model saved to {diabetes_model_path}")
    
    # Train prediabetes model on non-diabetic individuals
    if 'prediabetes' in df.columns:
        print("\nTraining prediabetes model...")
        # Filter to non-diabetic individuals
        nondiabetic_df = df[df['diabetes'] == 0]
        X_prediabetes = nondiabetic_df[all_features]
        y_prediabetes = nondiabetic_df['prediabetes']
        
        # Fill missing values
        for col in X_prediabetes.columns:
            if X_prediabetes[col].dtype in [np.float64, np.int64]:
                X_prediabetes[col] = X_prediabetes[col].fillna(X_prediabetes[col].mean())
            else:
                X_prediabetes[col] = X_prediabetes[col].fillna(X_prediabetes[col].mode()[0])
        
        # Train model
        prediabetes_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        prediabetes_model.fit(X_prediabetes, y_prediabetes)
        
        # Evaluate
        y_pred_prediabetes = prediabetes_model.predict(X_prediabetes)
        prediabetes_accuracy = accuracy_score(y_prediabetes, y_pred_prediabetes)
        print(f"Prediabetes model accuracy: {prediabetes_accuracy:.4f}")
        
        # Save model
        joblib.dump(prediabetes_model, diabetes_type_model_path)
        print(f"Prediabetes model saved to {diabetes_type_model_path}")
    else:
        print("\nWARNING: 'prediabetes' column not found, skipping prediabetes model")
    
    print("\nTraining completed successfully!")
    return True

if __name__ == "__main__":
    success = train_simplified_models()
    if success:
        print("\nModels are now ready for use in the Diabetes Prediction System.")
    else:
        print("\nERROR: Training failed. Please check the messages above.")
