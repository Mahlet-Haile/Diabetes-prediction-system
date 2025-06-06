"""
NHANES Data Integration
This module integrates the NHANES dataset with the Diabetes Prediction System
by providing data conversion utilities and model selection helpers.
"""
import os
import sys
import pandas as pd
import numpy as np
import json
from django.conf import settings

# Import the simplified NHANES model
from .nhanes_ml_models_simplified import NHANESDiabetesPredictionModel, nhanes_model as global_nhanes_model

class DiabetesPredictionIntegrator:
    """Class to use NHANES models exclusively for diabetes prediction"""
    
    def __init__(self, use_nhanes=True):
        """Initialize the integrator with NHANES models only"""
        # Always use NHANES models, no fallback to original
        self.use_nhanes = True
        
        # Get model paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        saved_models_dir = os.path.join(base_dir, 'DiabetesPrediction', 'saved_models')
        
        # NHANES model paths
        nhanes_diabetes_model_path = os.path.join(saved_models_dir, 'diabetes_model_nhanes.joblib')
        nhanes_type_model_path = os.path.join(saved_models_dir, 'diabetes_type_model_nhanes.joblib')
        
        # Initialize only NHANES model - use global instance if available
        if global_nhanes_model is not None:
            self.nhanes_model = global_nhanes_model
        else:
            self.nhanes_model = NHANESDiabetesPredictionModel()
        
        # Check if models exist
        self.nhanes_models_exist = os.path.exists(nhanes_diabetes_model_path) and os.path.exists(nhanes_type_model_path)
        
        if not self.nhanes_models_exist:
            import logging
            logging.error("NHANES models not found. Please train them using the Train Model button.")
    
    def get_active_model(self):
        """Return the NHANES model (no fallback to original)"""
        return self.nhanes_model
    
    def predict(self, data):
        """
        Make prediction using the active model
        
        Args:
            data: Dictionary or DataFrame with patient data
            
        Returns:
            Dictionary with prediction results
        """
        import json
        active_model = self.get_active_model()
        
        # Call predict_diabetes for simplified model
        if hasattr(active_model, 'predict_diabetes'):
            raw_prediction = active_model.predict_diabetes(data)
            
            # Convert to expected format for integration with expert system
            prediction = {
                'diabetes': raw_prediction.get('has_diabetes', False),
                'prediction_probability': raw_prediction.get('diabetes_probability', 0.0),
                'confidence_score': raw_prediction.get('diabetes_probability', 0.0),
                'risk_score': 0,  # Will be set by expert system
                'diagnosis': '',   # Will be set by expert system
                'diabetes_type': 'Type 2' if raw_prediction.get('has_diabetes', False) else 
                                ('Prediabetes' if raw_prediction.get('has_prediabetes', False) else 'None'),
                'recommendations': {}
            }
            return prediction
        else:
            # Fallback to original predict method
            return active_model.predict(data)
    
    def convert_assessment_to_nhanes(self, assessment):
        """
        Convert DiabetesAssessment model instance to NHANES-compatible format
        
        Args:
            assessment: DiabetesAssessment model instance
            
        Returns:
            Dictionary with NHANES-compatible fields
        """
        nhanes_data = {
            # Demographics
            'age': assessment.age,
            'gender': 1 if assessment.gender == 'M' else 2,  # 1=Male, 2=Female in NHANES
            
            # Anthropometrics
            'height': assessment.height,
            'weight': assessment.weight,
            'bmi': assessment.bmi,
            
            # Clinical measurements
            'blood_pressure_systolic': assessment.blood_pressure_systolic,
            'blood_pressure_diastolic': assessment.blood_pressure_diastolic,
            'fasting_glucose': assessment.fasting_glucose,
            # 'glucose' field removed as it's not in the NHANES dataset
            'hba1c': assessment.hba1c,
            'cholesterol': assessment.cholesterol,
            'hdl_cholesterol': assessment.hdl_cholesterol,  # New field for comprehensive assessment
            
            # Risk factors
            'smoking_history': self._convert_smoking_history(assessment),
            'smoking': 1 if assessment.smoking else 0,
            'alcohol': 1 if assessment.alcohol else 0,
            'active': 1 if assessment.active else 0,
            'physical_activity': assessment.physical_activity or 0,
            'family_history': 1 if assessment.family_history else 0,
            'hypertension': 1 if assessment.hypertension else 0,
            'heart_disease': 1 if assessment.heart_disease else 0,
            
            # Symptoms
            'polyuria': 1 if assessment.polyuria else 0,
            'polydipsia': 1 if assessment.polydipsia else 0,
            'polyphagia': 1 if assessment.polyphagia else 0,
            'weight_loss': 1 if assessment.weight_loss else 0,
            'fatigue': 1 if assessment.fatigue else 0,
            'blurred_vision': 1 if assessment.blurred_vision else 0,
            'slow_healing': 1 if assessment.slow_healing else 0,
            'tingling': 1 if assessment.tingling else 0,
            
            # New symptoms for comprehensive assessment
            'skin_darkening': 1 if assessment.skin_darkening else 0,
            'frequent_infections': 1 if assessment.frequent_infections else 0,
            
            # Complications
            'chest_pain': 1 if assessment.chest_pain else 0,
            'shortness_of_breath': 1 if assessment.shortness_of_breath else 0,
            'swelling_in_legs': 1 if assessment.swelling_in_legs else 0,
            'numbness': 1 if assessment.numbness else 0,
            'foot_ulcers': 1 if assessment.foot_ulcers else 0,
            'vision_loss': 1 if assessment.vision_loss else 0,
        }
        
        return nhanes_data
    
    def _convert_smoking_history(self, assessment):
        """Convert smoking history from assessment to NHANES format (binary 0/1)"""
        if not hasattr(assessment, 'smoking_history') or not assessment.smoking_history:
            return 0  # Default to not smoker (0) if no information
        
        # Map simplified smoking history to binary (0/1) for NHANES format
        smoking_map = {
            'not_smoker': 0,  # Not a smoker -> 0
            'smoker': 1,      # Smoker -> 1
            'no_info': 0       # Default to not smoker if no info
        }
        
        return smoking_map.get(assessment.smoking_history, 0)
    
    def update_assessment_with_prediction(self, assessment, prediction_result):
        """
        Update DiabetesAssessment instance with prediction results
        
        Args:
            assessment: DiabetesAssessment model instance
            prediction_result: Dictionary with prediction results
            
        Returns:
            Updated DiabetesAssessment instance
        """
        # Update basic prediction fields
        assessment.diabetes = prediction_result['diabetes']
        assessment.prediction_probability = prediction_result['prediction_probability']
        assessment.diabetes_type = prediction_result['diabetes_type']
        assessment.confidence_score = prediction_result['confidence_score']
        assessment.risk_score = prediction_result['risk_score']
        
        # Update expert system results
        assessment.diagnosis = prediction_result['diagnosis']
        
        # Update recommendations as JSON
        if 'recommendations' in prediction_result:
            assessment.recommendations = prediction_result['recommendations']
            
            # Also update legacy recommendation fields for backward compatibility
            if 'diet' in prediction_result['recommendations']:
                diet_recs = prediction_result['recommendations']['diet']
                assessment.diet_recommendations = json.dumps(diet_recs) if isinstance(diet_recs, list) else '[]'
            
            # Map 'activity' from expert system to 'exercise_recommendations' in the model
            if 'activity' in prediction_result['recommendations']:
                activity_recs = prediction_result['recommendations']['activity']
                assessment.exercise_recommendations = json.dumps(activity_recs) if isinstance(activity_recs, list) else '[]'
            
            # Map 'lifestyle' and 'medical' to 'monitoring_recommendations' in the model
            monitoring_texts = []
            
            if 'lifestyle' in prediction_result['recommendations']:
                lifestyle_recs = prediction_result['recommendations']['lifestyle']
                if isinstance(lifestyle_recs, list):
                    monitoring_texts.extend(lifestyle_recs)
                else:
                    monitoring_texts.append(str(lifestyle_recs))
                    
            if 'medical' in prediction_result['recommendations']:
                medical_recs = prediction_result['recommendations']['medical']
                if isinstance(medical_recs, list):
                    monitoring_texts.extend(medical_recs)
                else:
                    monitoring_texts.append(str(medical_recs))
                    
            if monitoring_texts:
                assessment.monitoring_recommendations = '\n'.join(monitoring_texts)
        
        return assessment

# Helper functions for common use cases

def get_integrator(use_nhanes=True):
    """Get a configured integrator instance"""
    return DiabetesPredictionIntegrator(use_nhanes=use_nhanes)

def predict_with_assessment(assessment, use_nhanes=True):
    """
    Make prediction using an assessment instance
    
    Args:
        assessment: DiabetesAssessment model instance
        use_nhanes: Whether to use NHANES models
        
    Returns:
        Updated assessment with prediction results
    """
    integrator = get_integrator(use_nhanes=use_nhanes)
    nhanes_data = integrator.convert_assessment_to_nhanes(assessment)
    prediction = integrator.predict(nhanes_data)
    
    if prediction:
        assessment = integrator.update_assessment_with_prediction(assessment, prediction)
    
    return assessment
