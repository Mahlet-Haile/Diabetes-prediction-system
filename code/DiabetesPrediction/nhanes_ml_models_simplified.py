"""NHANES Diabetes Prediction Models - Simplified

This module provides a simplified interface to load and use the trained NHANES diabetes prediction models.
It removes the complex training functionality which is now handled by fix_model_training.py.
"""

import os
import joblib
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class NHANESDiabetesPredictionModel:
    """Machine learning model for diabetes prediction using NHANES data"""
    
    def __init__(self):
        """Initialize the model paths"""
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
        self.diabetes_model_path = os.path.join(self.model_path, 'diabetes_model_nhanes.joblib')
        self.prediabetes_model_path = os.path.join(self.model_path, 'diabetes_type_model_nhanes.joblib')
        self.diabetes_model = None
        self.prediabetes_model = None
    
    def _load_models(self):
        """Load the trained models from disk"""
        try:
            # Check if models exist
            if os.path.exists(self.diabetes_model_path) and os.path.exists(self.prediabetes_model_path):
                self.diabetes_model = joblib.load(self.diabetes_model_path)
                self.prediabetes_model = joblib.load(self.prediabetes_model_path)
                logger.info("Successfully loaded NHANES diabetes models")
                return True
            else:
                logger.warning(f"Models not found at {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def train_models(self, dataset_path=None):
        """
        Simplified training method that calls the dedicated training script
        This is kept for compatibility with existing code
        """
        try:
            # Import the training function from our fixed script
            from .fix_model_training import fix_models
            
            # Run the training
            result = fix_models()
            
            # Reload models if training was successful
            if result:
                self._load_models()
            
            return result
        except Exception as e:
            logger.error(f"Error in training models: {str(e)}")
            return False
    
    def predict_diabetes(self, patient_data):
        """
        Predict whether a patient has diabetes
        
        Args:
            patient_data (dict): Patient data with clinical measurements
            
        Returns:
            dict: Prediction results with probabilities
        """
        # Load models if not already loaded
        if self.diabetes_model is None or self.prediabetes_model is None:
            if not self._load_models():
                return {"error": "Models not loaded"}
        
        try:
            # Convert patient data to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Fill missing values (simple imputation)
            for col in df.columns:
                if col not in ['participant_id']:  # Skip ID columns
                    df[col] = df[col].fillna(0)
            
            # Make diabetes prediction
            diabetes_prob = self.diabetes_model.predict_proba(df)[0]
            diabetes_prediction = self.diabetes_model.predict(df)[0]
            
            result = {
                "has_diabetes": bool(diabetes_prediction),
                "diabetes_probability": float(diabetes_prob[1]),
            }
            
            # If no diabetes, check for prediabetes
            if not diabetes_prediction:
                prediabetes_prob = self.prediabetes_model.predict_proba(df)[0]
                prediabetes_prediction = self.prediabetes_model.predict(df)[0]
                
                result.update({
                    "has_prediabetes": bool(prediabetes_prediction),
                    "prediabetes_probability": float(prediabetes_prob[1]),
                })
            else:
                result.update({
                    "has_prediabetes": False,
                    "prediabetes_probability": 0.0,
                })
            
            return result
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {"error": str(e)}

# Create a global instance for easy access
nhanes_model = NHANESDiabetesPredictionModel()
