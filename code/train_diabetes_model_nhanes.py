"""
Script to train the diabetes prediction model using the NHANES dataset
"""
import os
import sys
import django
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Set up Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'FinalProject.settings')
django.setup()

# Import the model after Django setup
from DiabetesPrediction.nhanes_ml_models import NHANESDiabetesPredictionModel

def train_model_with_nhanes():
    """Train the diabetes prediction model using the NHANES dataset"""
    print("Initializing NHANES diabetes prediction model...")
    diabetes_model = NHANESDiabetesPredictionModel()
    
    # Path to the NHANES dataset
    nhanes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nhanes_data')
    dataset_path = os.path.join(nhanes_dir, 'nhanes_diabetes_dataset.csv')
    
    if not os.path.exists(dataset_path):
        print(f"Error: NHANES dataset not found at {dataset_path}")
        print("Please run the generate_realistic_nhanes.py script first to create the dataset.")
        return False
    
    print(f"Using NHANES dataset: {dataset_path}")
    
    # Count the number of records in the dataset
    try:
        df = pd.read_csv(dataset_path)
        record_count = len(df)
        diabetes_count = df['diabetes'].sum()
        prediabetes_count = df['prediabetes'].sum()
        normal_count = record_count - diabetes_count - prediabetes_count
        
        print(f"Dataset contains {record_count} records:")
        print(f"  - {diabetes_count} ({diabetes_count/record_count*100:.1f}%) diabetes cases")
        print(f"  - {prediabetes_count} ({prediabetes_count/record_count*100:.1f}%) prediabetes cases")
        print(f"  - {normal_count} ({normal_count/record_count*100:.1f}%) non-diabetic cases")
    except Exception as e:
        print(f"Warning: Could not analyze dataset: {e}")
    
    # Train the model with the NHANES dataset
    # Use the train_models method directly from our NHANESDiabetesPredictionModel class
    result = diabetes_model.train_models(dataset_path)
    
    if result:
        print("Model training completed successfully!")
        return True
    else:
        print("Error training the model.")
        return False

# This function is no longer used as we now use the train_models method in NHANESDiabetesPredictionModel
def _unused_train_models_with_nhanes(dataset_path):
    """
    Train both diabetes detection and type prediction models using the NHANES dataset
    
    Args:
        dataset_path: Path to the NHANES dataset CSV file
        
    Returns:
        Dictionary with training results and accuracies
    """
    try:
        # Load and preprocess data
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(df)} records")
        
        # Map NHANES features to the expected model features
        print("Mapping NHANES features to model features...")
        
        # Create feature mapping for required fields
        df_processed = df.copy()
        
        # Map gender (1=Male, 2=Female in NHANES) - keep as numeric for ML models
        # We'll keep original gender codes: 1=Male, 2=Female
        df_processed['gender_code'] = df_processed['gender']
        
        # Map blood glucose level
        if 'fasting_glucose' in df.columns:
            df_processed['blood_glucose_level'] = df_processed['fasting_glucose']
        elif 'glucose' in df.columns:
            df_processed['blood_glucose_level'] = df_processed['glucose']
        
        # Map HbA1c
        if 'hba1c' in df.columns:
            df_processed['hba1c_level'] = df_processed['hba1c']
        
        # Map symptoms to model feature names if they exist
        symptom_mapping = {
            'polyuria': 'frequent_urination',
            'polydipsia': 'excessive_thirst',
            'weight_loss': 'unexplained_weight_loss',
            'fatigue': 'fatigue',
            'blurred_vision': 'blurred_vision',
            'slow_healing': 'slow_healing_sores',
            'tingling': 'numbness_tingling'
        }
        
        for nhanes_name, model_name in symptom_mapping.items():
            if nhanes_name in df.columns:
                df_processed[model_name] = df_processed[nhanes_name]
            else:
                # Create default values if the column doesn't exist
                df_processed[model_name] = 0
        
        # Make sure all required features exist
        required_features = [
            'age', 'bmi', 'blood_glucose_level', 'hba1c_level', 
            'hypertension', 'heart_disease', 'gender',
            'excessive_thirst', 'frequent_urination', 
            'unexplained_weight_loss', 'fatigue', 
            'blurred_vision', 'slow_healing_sores', 
            'numbness_tingling'
        ]
        
        for feature in required_features:
            if feature not in df_processed.columns:
                print(f"Warning: Feature '{feature}' not found in dataset. Creating default values.")
                if feature in ['age', 'bmi', 'blood_glucose_level', 'hba1c_level']:
                    # For numerical features, use median values
                    df_processed[feature] = 0  # Will be filled with median later
                else:
                    # For categorical or binary features
                    df_processed[feature] = 0
        
        # Fill missing values with appropriate defaults
        for col in ['bmi', 'blood_glucose_level', 'hba1c_level']:
            if df_processed[col].isnull().any():
                median_value = df_processed[col].median()
                df_processed[col].fillna(median_value, inplace=True)
        
        if 'smoking_history' not in df_processed.columns:
            df_processed['smoking_history'] = 'no_info'
        
        # Encode categorical variables
        smoking_map = {
            'never': 'never', 
            'former': 'former', 
            'current': 'current', 
            'ever': 'ever',
            'no_info': 'no_info'
        }
        df_processed['smoking_history'] = df_processed['smoking_history'].map(
            lambda x: smoking_map.get(x, 'no_info')
        )
        
        # Create the diabetes type column based on diabetes and prediabetes flags
        # Types: 'none', 'prediabetes', 'type1', 'type2'
        df_processed['diabetes_type'] = 'none'
        
        # Prediabetes cases
        prediabetes_mask = (df_processed['prediabetes'] == 1) & (df_processed['diabetes'] == 0)
        df_processed.loc[prediabetes_mask, 'diabetes_type'] = 'prediabetes'
        
        # For diabetes cases, we need to determine type1 vs type2
        # In real data, we'd have this information directly
        # For now, we'll use a heuristic: people under 20 with low BMI are more likely type1
        diabetes_mask = df_processed['diabetes'] == 1
        type1_indicators = (df_processed['age'] < 20) | (df_processed['bmi'] < 25)
        
        type1_mask = diabetes_mask & type1_indicators
        type2_mask = diabetes_mask & (~type1_indicators)
        
        df_processed.loc[type1_mask, 'diabetes_type'] = 'type1'
        df_processed.loc[type2_mask, 'diabetes_type'] = 'type2'
        
        print(f"Dataset prepared with all required features")
        
        # Make sure categorical features are numeric
        # Handle any string columns
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                print(f"Converting string column '{col}' to numeric")
                # For diabetes_type, we'll keep it as-is since it's a target
                if col != 'diabetes_type':
                    # Use one-hot encoding for categorical variables
                    df_processed = pd.get_dummies(df_processed, columns=[col], drop_first=True)
        
        # Split into features and targets
        X = df_processed.drop(['diabetes', 'prediabetes', 'diabetes_type', 'participant_id', 'gender'], axis=1, errors='ignore')
        y_diabetes = df_processed['diabetes']
        y_type = df_processed['diabetes_type']
        
        # Split into training and testing sets
        X_train, X_test, y_train_diabetes, y_test_diabetes = train_test_split(
            X, y_diabetes, test_size=0.2, random_state=42
        )
        
        # Also split the type target
        _, _, y_train_type, y_test_type = train_test_split(
            X, y_type, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} records, Testing set: {len(X_test)} records")
        
        # Create and train the diabetes detection model
        print("Training diabetes detection model...")
        diabetes_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        diabetes_model.fit(X_train, y_train_diabetes)
        
        # Evaluate the diabetes detection model
        y_pred_diabetes = diabetes_model.predict(X_test)
        diabetes_accuracy = accuracy_score(y_test_diabetes, y_pred_diabetes)
        
        print(f"Diabetes detection model accuracy: {diabetes_accuracy:.4f}")
        print("\nClassification Report (Diabetes Detection):")
        print(classification_report(y_test_diabetes, y_pred_diabetes))
        
        # Create and train the diabetes type model
        print("\nTraining diabetes type model...")
        type_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
        
        type_model.fit(X_train, y_train_type)
        
        # Evaluate the diabetes type model
        y_pred_type = type_model.predict(X_test)
        type_accuracy = accuracy_score(y_test_type, y_pred_type)
        
        print(f"Diabetes type model accuracy: {type_accuracy:.4f}")
        print("\nClassification Report (Diabetes Type):")
        print(classification_report(y_test_type, y_pred_type))
        
        # Save the models
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  'DiabetesPrediction', 'saved_models')
        os.makedirs(model_path, exist_ok=True)
        
        diabetes_model_path = os.path.join(model_path, 'diabetes_model_nhanes.joblib')
        type_model_path = os.path.join(model_path, 'diabetes_type_model_nhanes.joblib')
        
        joblib.dump(diabetes_model, diabetes_model_path)
        joblib.dump(type_model, type_model_path)
        
        print(f"Models saved to {model_path}")
        
        return {
            'diabetes_accuracy': diabetes_accuracy,
            'type_accuracy': type_accuracy,
            'diabetes_model_path': diabetes_model_path,
            'type_model_path': type_model_path
        }
        
    except Exception as e:
        print(f"Error training models: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting model training with NHANES dataset...")
    success = train_model_with_nhanes()
    if success:
        print("Training complete! The NHANES models are now ready for predictions.")
    else:
        print("Training failed. Please check the error messages above.")
