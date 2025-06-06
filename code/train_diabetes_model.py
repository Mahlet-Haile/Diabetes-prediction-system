"""
Script to train the diabetes prediction model
"""
import os
import sys
import django
import pandas as pd

# Set up Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'FinalProject.settings')
django.setup()

# Import the model after Django setup
from DiabetesPrediction.ml_models import DiabetesPredictionModel

def train_model():
    """Train the diabetes prediction model using the expanded dataset for better accuracy"""
    print("Initializing diabetes prediction model...")
    diabetes_model = DiabetesPredictionModel()
    
    # Try multiple possible paths for the datasets
    base_dirs = [
        # Relative path from script directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
        # Direct path from code directory
        os.path.dirname(os.path.abspath(__file__)),
        # Absolute path to Desktop/GDPS
        r'C:\Users\hi\Desktop\GDPS'
    ]
    
    # Find the correct path to the dataset files
    expanded_dataset_path = None
    original_dataset_path = None
    
    for base_dir in base_dirs:
        exp_path = os.path.join(base_dir, 'data', 'diabetes_prediction_dataset_expanded.csv')
        orig_path = os.path.join(base_dir, 'data', 'diabetes_prediction_dataset.csv')
        
        print(f"Checking for dataset in: {base_dir}")
        
        if os.path.exists(exp_path):
            expanded_dataset_path = exp_path
            print(f"Found expanded dataset: {exp_path}")
            
        if os.path.exists(orig_path):
            original_dataset_path = orig_path
            print(f"Found original dataset: {orig_path}")
            
        if expanded_dataset_path or original_dataset_path:
            break
    
    # Decide which dataset to use
    if os.path.exists(expanded_dataset_path):
        dataset_path = expanded_dataset_path
        print(f"Using expanded dataset with 10,000 records for better model accuracy")
    else:
        dataset_path = original_dataset_path
        print(f"Expanded dataset not found. Using original dataset with limited records.")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return False
    
    # Count the number of records in the dataset
    try:
        df = pd.read_csv(dataset_path)
        record_count = len(df)
        print(f"Training model with dataset: {dataset_path} ({record_count} records)")
    except Exception as e:
        print(f"Warning: Could not count records in dataset: {e}")
        print(f"Training model with dataset: {dataset_path}")
    
    # Train the model
    result = diabetes_model.train_models(dataset_path)
    
    if result:
        print("Model training completed successfully!")
        print(f"Model accuracy: {result['accuracy']:.4f}")
        return True
    else:
        print("Error training the model.")
        return False

if __name__ == "__main__":
    print("Starting model training...")
    success = train_model()
    if success:
        print("Training complete! The model is now ready for predictions.")
    else:
        print("Training failed. Please check the error messages above.")
