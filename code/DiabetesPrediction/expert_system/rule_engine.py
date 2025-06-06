import json

"""
Rule Engine for Diabetes Expert System

This module implements the inference engine and rule processing for the diabetes expert system.
It analyzes patient data, applies medical knowledge rules, and generates personalized assessments.
"""

from . import knowledge_base as kb
import numpy as np

# Helper function for risk level calculation
def get_risk_level(risk_score):
    """Convert a numeric risk score to a risk level category"""
    if risk_score <= 0:
        return 'none'
    elif risk_score <= 1:
        return 'low'
    elif risk_score <= 3:
        return 'moderate'
    else:
        return 'high'

class DiabetesExpertSystem:
    """Expert system for diabetes diagnosis, risk assessment, and recommendations."""
    
    def __init__(self, patient_data=None):
        """Initialize the expert system with patient data."""
        self.patient_data = patient_data or {}
        self.risk_score = 0
        self.diagnosis = None
        self.explanation = []
        self.recommendations = {
            'diet': [],
            'activity': [],
            'lifestyle': [],
            'medical': []
        }
        self.complication_risks = {}
        
    def load_patient_data(self, patient_data):
        """Load patient data into the expert system."""
        self.patient_data = patient_data
        
    def run_assessment(self):
        """
        Run a full diabetes assessment using the expert system.
        This is the main method that should be called from external code.
        
        Returns:
            dict: Dictionary containing diagnosis, risk score, and recommendations
        """
        # Run risk assessment (this also determines diagnosis)
        self.assess_diabetes_risk()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Check for complications if diagnosed with diabetes
        if self.diagnosis == 'diabetes':
            self.assess_complication_risks()
        
        # Ensure recommendations are properly structured
        recommendations = {
            'diet': self.recommendations.get('diet', []),
            'activity': self.recommendations.get('activity', []),
            'lifestyle': self.recommendations.get('lifestyle', []),
            'medical': self.recommendations.get('medical', [])
        }
        
        # Return results in a structured format
        return {
            'diagnosis': self.diagnosis,
            'risk_score': self.risk_score,
            'recommendations': recommendations,
            'explanation': self.explanation,
            'complication_risks': self.complication_risks
        }
        
    def assess_diabetes_risk(self):
        """
        Assess the patient's diabetes risk based on risk factors, 
        symptoms, and clinical measurements.
        """
        self.risk_score = 0
        self.explanation = []
        
        # Process age as a risk factor
        if 'age' in self.patient_data:
            age = self.patient_data['age']
            if age >= kb.RISK_FACTORS['age']['high']:
                self.risk_score += kb.RISK_FACTORS['age']['weight']
                self.explanation.append(f"Age {age} years is a risk factor (≥45 years)")
        
        # Process BMI as a risk factor
        if 'bmi' in self.patient_data:
            bmi = self.patient_data['bmi']
            if bmi >= kb.RISK_FACTORS['bmi']['obese']:
                self.risk_score += kb.RISK_FACTORS['bmi']['weight'] * 2
                self.explanation.append(f"BMI of {bmi:.1f} indicates obesity, a significant risk factor")
            elif bmi >= kb.RISK_FACTORS['bmi']['overweight']:
                self.risk_score += kb.RISK_FACTORS['bmi']['weight']
                self.explanation.append(f"BMI of {bmi:.1f} indicates overweight, a risk factor")
        
        # Process family history
        if 'family_history' in self.patient_data and self.patient_data['family_history']:
            self.risk_score += kb.RISK_FACTORS['family_history']['weight']
            self.explanation.append("Family history of diabetes increases risk")
        
        # Process hypertension
        if 'hypertension' in self.patient_data and self.patient_data['hypertension']:
            self.risk_score += kb.RISK_FACTORS['hypertension']['weight']
            self.explanation.append("Presence of hypertension increases risk")
        
        # Process physical activity
        if 'physical_activity' in self.patient_data and self.patient_data['physical_activity'] is not None:
            if self.patient_data['physical_activity'] < kb.RISK_FACTORS['physical_activity']['low']:
                self.risk_score += kb.RISK_FACTORS['physical_activity']['weight']
                self.explanation.append("Low physical activity increases risk")
        
        # Process smoking status
        if 'smoking' in self.patient_data and self.patient_data['smoking']:
            self.risk_score += kb.RISK_FACTORS['smoking']['weight']
            self.explanation.append("Smoking increases diabetes risk")
            
        # Process symptoms with diminishing returns for overlapping symptoms
        # Track symptom clusters for applying diminishing returns
        symptom_clusters = {
            'polyuria_polydipsia_polyphagia': {
                'symptoms': ['polyuria', 'polydipsia', 'polyphagia'],
                'present': [],
                'count': 0,
                'explanation': "These symptoms (excessive urination, thirst, and hunger) often occur together due to the same underlying hyperglycemia"
            }
        }
        
        # First pass: identify which symptoms are present and which clusters they belong to
        for symptom, info in kb.SYMPTOMS.items():
            if symptom in self.patient_data and self.patient_data[symptom]:
                # Add the base explanation for the symptom
                self.explanation.append(f"Presence of {symptom.replace('_', ' ')} ({info['description']}) is consistent with diabetes")
                
                # Check if this symptom belongs to a cluster
                for cluster_name, cluster in symptom_clusters.items():
                    if symptom in cluster['symptoms']:
                        cluster['present'].append(symptom)
                        cluster['count'] += 1
        
        # Second pass: apply standard weights for non-clustered symptoms and diminishing returns for clustered ones
        processed_symptoms = []
        
        # Process clusters with diminishing returns
        for cluster_name, cluster in symptom_clusters.items():
            # Skip if no symptoms in this cluster are present
            if cluster['count'] == 0:
                continue
                
            # Apply diminishing returns based on how many symptoms in the cluster are present
            if cluster['count'] == 1:
                # For a single symptom, apply full weight
                symptom = cluster['present'][0]
                self.risk_score += kb.SYMPTOMS[symptom]['weight']
                processed_symptoms.append(symptom)
            elif cluster['count'] > 1:
                # For multiple symptoms, apply diminishing returns
                # First symptom gets full weight
                base_weight = kb.SYMPTOMS[cluster['present'][0]]['weight']
                self.risk_score += base_weight
                
                # Additional symptoms get reduced weights (70% for second, 30% for third)
                if cluster['count'] >= 2:
                    second_symptom_weight = kb.SYMPTOMS[cluster['present'][1]]['weight'] * 0.7
                    self.risk_score += second_symptom_weight
                    
                if cluster['count'] >= 3:
                    third_symptom_weight = kb.SYMPTOMS[cluster['present'][2]]['weight'] * 0.3
                    self.risk_score += third_symptom_weight
                
                # Add an explanation about the diminishing returns applied
                if cluster['count'] >= 2:
                    self.explanation.append(f"Note: Applied diminishing returns to related symptoms: {', '.join([s.replace('_', ' ') for s in cluster['present']])}. {cluster['explanation']}")
                
                # Mark all symptoms in this cluster as processed
                processed_symptoms.extend(cluster['present'])
        
        # Process remaining symptoms with standard weights
        for symptom, info in kb.SYMPTOMS.items():
            if symptom in self.patient_data and self.patient_data[symptom] and symptom not in processed_symptoms:
                self.risk_score += info['weight']
                # Explanation already added in first pass
                
        # Process blood glucose measurements if available
        if 'fasting_glucose' in self.patient_data:
            fg = self.patient_data['fasting_glucose']
            if fg >= kb.DIAGNOSIS_THRESHOLDS['fasting_glucose']['diabetes']:
                self.risk_score += 10
                self.explanation.append(f"Fasting glucose of {fg} mg/dL is consistent with diabetes (≥126 mg/dL)")
            elif fg >= kb.DIAGNOSIS_THRESHOLDS['fasting_glucose']['prediabetes']:
                self.risk_score += 5
                self.explanation.append(f"Fasting glucose of {fg} mg/dL indicates prediabetes (100-125 mg/dL)")
                
        if 'hba1c' in self.patient_data:
            hba1c = self.patient_data['hba1c']
            if hba1c >= kb.DIAGNOSIS_THRESHOLDS['hba1c']['diabetes']:
                self.risk_score += 10
                self.explanation.append(f"HbA1c of {hba1c}% is consistent with diabetes (≥6.5%)")
            elif hba1c >= kb.DIAGNOSIS_THRESHOLDS['hba1c']['prediabetes']:
                self.risk_score += 5
                self.explanation.append(f"HbA1c of {hba1c}% indicates prediabetes (5.7-6.4%)")
        
        # Determine diagnosis category based on clinical values and risk score
        # Helper function to safely convert values to numeric for comparison
        def safe_numeric(value):
            if value is None:
                return 0
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                # Skip text values like 'normal', 'high', etc.
                if value.replace('.', '', 1).isdigit():
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return 0
            return 0
            
        # Check for diabetes based on clinical measurements with appropriate messaging and risk score adjustment
        if 'fasting_glucose' in self.patient_data and safe_numeric(self.patient_data['fasting_glucose']) >= kb.DIAGNOSIS_THRESHOLDS['fasting_glucose']['diabetes']:
            self.diagnosis = 'diabetes'
            self.explanation.append(f"Fasting glucose of {self.patient_data['fasting_glucose']} mg/dL meets clinical criteria for diabetes (≥126 mg/dL)")
            # Ensure risk score is appropriately high for diagnosed diabetes
            if self.risk_score < 26:
                self.risk_score = 26
        elif 'hba1c' in self.patient_data and safe_numeric(self.patient_data['hba1c']) >= kb.DIAGNOSIS_THRESHOLDS['hba1c']['diabetes']:
            self.diagnosis = 'diabetes'
            self.explanation.append(f"HbA1c of {self.patient_data['hba1c']}% meets clinical criteria for diabetes (≥6.5%)")
            # Ensure risk score is appropriately high for diagnosed diabetes
            if self.risk_score < 26:
                self.risk_score = 26
        # Random glucose diagnostic check removed as it's not in the NHANES dataset
        
        # Then check for prediabetes based on clinical measurements
        elif 'fasting_glucose' in self.patient_data and safe_numeric(self.patient_data['fasting_glucose']) >= kb.DIAGNOSIS_THRESHOLDS['fasting_glucose']['prediabetes']:
            self.diagnosis = 'prediabetes'
            self.explanation.append(f"Fasting glucose of {self.patient_data['fasting_glucose']} mg/dL meets clinical criteria for prediabetes (100-125 mg/dL)")
            # Ensure risk score is appropriately high for diagnosed prediabetes
            if self.risk_score < 20:
                self.risk_score = 20
        elif 'hba1c' in self.patient_data and safe_numeric(self.patient_data['hba1c']) >= kb.DIAGNOSIS_THRESHOLDS['hba1c']['prediabetes']:
            self.diagnosis = 'prediabetes'
            self.explanation.append(f"HbA1c of {self.patient_data['hba1c']}% meets clinical criteria for prediabetes (5.7-6.4%)")
            # Ensure risk score is appropriately high for diagnosed prediabetes
            if self.risk_score < 20:
                self.risk_score = 20
        # Random glucose prediabetes diagnostic check removed as it's not in the NHANES dataset
        
        # If no definitive diagnosis from clinical measurements, use risk score
        else:
            # Apply new clearer risk thresholds
            if self.risk_score >= 26:
                self.diagnosis = 'very_high_risk'
                self.explanation.append(f"Risk score of {self.risk_score:.1f} indicates very high risk of developing diabetes (score ≥ 26)")
            elif self.risk_score >= 20:
                self.diagnosis = 'high_risk'
                self.explanation.append(f"Risk score of {self.risk_score:.1f} indicates high risk of developing diabetes (score 20-25)")
            elif self.risk_score >= 10:
                self.diagnosis = 'moderate_risk'
                self.explanation.append(f"Risk score of {self.risk_score:.1f} indicates moderate risk of developing diabetes (score 10-19)")
            else:
                self.diagnosis = 'low_risk'
                self.explanation.append(f"Risk score of {self.risk_score:.1f} indicates low risk of developing diabetes (score 0-9)")
            
        return {
            'risk_score': self.risk_score,
            'diagnosis': self.diagnosis,
            'explanation': self.explanation
        }
    
    def generate_recommendations(self):
        """Generate personalized recommendations based on patient data and diagnosis."""
        # Determine which category of recommendations to use
        if self.diagnosis == 'diabetes':
            diet_key = 'diabetes'
            activity_key = 'diabetes'
            lifestyle_key = 'diabetes'
        elif self.diagnosis == 'prediabetes' or self.diagnosis == 'high_risk':
            diet_key = 'prediabetes'
            activity_key = 'prediabetes'
            lifestyle_key = 'prediabetes'
        else:
            diet_key = 'normal'
            activity_key = 'normal' 
            lifestyle_key = 'normal'
            
        # Get general recommendations based on diagnosis
        self.recommendations['diet'] = kb.DIET_RECOMMENDATIONS[diet_key]
        self.recommendations['activity'] = kb.ACTIVITY_RECOMMENDATIONS[activity_key]
        self.recommendations['lifestyle'] = kb.LIFESTYLE_RECOMMENDATIONS[lifestyle_key]
        
        # Add personalized medical recommendations based on diagnosis and risk level
        if self.diagnosis == 'diabetes':
            self.recommendations['medical'] = [
                "Schedule regular follow-ups every 3-6 months with your healthcare provider",
                "Get recommended lab tests for diabetes monitoring including HbA1c every 3-6 months",
                "Have annual comprehensive eye exams with an ophthalmologist",
                "Get annual foot exams and check your feet daily for cuts, sores, and changes",
                "Monitor your blood pressure and cholesterol levels regularly",
                "Have kidney function tests at least once a year (urine albumin and serum creatinine)",
                "Discuss medication options with your healthcare provider including their benefits and potential side effects",
                "Ask about whether a continuous glucose monitor (CGM) or insulin pump might benefit you",
                "Consider working with a Certified Diabetes Educator to develop self-management skills"
            ]
            
            # Add personalized recommendations based on patient data
            if 'bmi' in self.patient_data and self.patient_data['bmi'] >= kb.RISK_FACTORS['bmi']['overweight']:
                target_weight_loss = round(self.patient_data['weight'] * 0.07, 1)
                self.recommendations['medical'].append(
                    f"Discuss weight management strategies with your healthcare provider - a weight loss of {target_weight_loss} kg (7% of body weight) can significantly improve blood glucose control"
                )
                
            if 'hypertension' in self.patient_data and self.patient_data['hypertension']:
                self.recommendations['medical'].append(
                    "Work with your healthcare provider to keep blood pressure below 130/80 mmHg to reduce risk of complications"
                )
                
            if 'smoking' in self.patient_data and self.patient_data['smoking']:
                self.recommendations['medical'].append(
                    "Discuss smoking cessation strategies with your healthcare provider - quitting smoking is one of the most important steps to reduce diabetes complications"
                )
                
        elif self.diagnosis == 'prediabetes' or self.diagnosis == 'high_risk':
            self.recommendations['medical'] = [
                "Schedule a follow-up appointment with your healthcare provider to discuss personalized diabetes prevention strategies",
                "Get screened for diabetes every 6-12 months with an HbA1c or fasting glucose test",
                "Have your blood pressure and cholesterol levels checked regularly",
                "Discuss whether medications like metformin might be appropriate for diabetes prevention",
                "Ask about referral to a CDC-recognized Diabetes Prevention Program",
                "Consider consulting with a registered dietitian for personalized nutrition guidance"
            ]
            
            # Add personalized recommendations based on risk factors
            if 'bmi' in self.patient_data and self.patient_data['bmi'] >= kb.RISK_FACTORS['bmi']['overweight']:
                target_weight_loss = round(self.patient_data['weight'] * 0.07, 1)
                self.recommendations['medical'].append(
                    f"Discuss strategies to achieve a weight loss goal of {target_weight_loss} kg (7% of body weight), which can reduce diabetes risk by up to 58%"
                )
            
        elif self.diagnosis == 'moderate_risk':
            self.recommendations['medical'] = [
                "Schedule annual check-ups with your healthcare provider",
                "Get screened for diabetes and cardiovascular risk factors annually",
                "Discuss family history of diabetes with your healthcare provider",
                "Consider a fasting blood glucose test or HbA1c screening",
                "Have your blood pressure checked regularly"
            ]
            
        elif self.diagnosis == 'low_risk':
            self.recommendations['medical'] = [
                "Continue with regular health check-ups",
                "Consider getting screened for diabetes every 3 years, especially if risk factors are present",
                "Know your family history of diabetes and share it with your healthcare provider",
                "Discuss any symptoms like increased thirst, urination, or unexplained weight loss with your doctor"
            ]
            
        else:  # normal risk
            self.recommendations['medical'] = [
                "Continue with regular health check-ups",
                "Consider getting screened for diabetes every 3 years after age 45",
                "Discuss any new risk factors with your healthcare provider",
                "Stay informed about the signs and symptoms of diabetes"
            ]
            
        # Add symptom-specific recommendations
        if any(symptom in self.patient_data and self.patient_data[symptom] for symptom in ['polyuria', 'polydipsia', 'polyphagia']):
            self.recommendations['medical'].append("Your symptoms of excessive urination, thirst, or hunger should be discussed with a healthcare provider promptly")
            
        return self.recommendations
    
    def assess_complication_risks(self):
        """Assess the patient's risk for various diabetes complications."""
        # Only assess complications for diabetes diagnosis
        if self.diagnosis != 'diabetes':
            return {}
            
        for complication, data in kb.COMPLICATIONS.items():
            risk_score = 0
            risk_factors_present = []
            
            # Check if patient has any risk factors for this complication
            for risk_factor in data['risk_factors']:
                if risk_factor in self.patient_data and self.patient_data[risk_factor]:
                    risk_score += 1
                    risk_factors_present.append(risk_factor.replace('_', ' '))
            
            # Check if patient has any warning signs for this complication
            warning_signs_present = []
            for sign in data['warning_signs']:
                if sign in self.patient_data and self.patient_data[sign]:
                    risk_score += 2  # Warning signs are weighted higher
                    warning_signs_present.append(sign.replace('_', ' '))
            
            # Calculate risk level
            risk_level = get_risk_level(risk_score)
                
            # Store complication risk assessment
            self.complication_risks[complication] = {
                'risk_level': risk_level,
                'risk_factors': risk_factors_present,
                'warning_signs': warning_signs_present,
                'recommendations': data['recommendations']
            }
            
        return self.complication_risks
    
    def get_medication_info(self, medication_name):
        """Get information about a specific diabetes medication."""
        if medication_name in kb.MEDICATIONS:
            return kb.MEDICATIONS[medication_name]
        return None
    
    def process_assessment(self):
        """Process a complete diabetes assessment and return results."""
        risk_assessment = self.assess_diabetes_risk()
        recommendations = self.generate_recommendations()
        
        if self.diagnosis == 'diabetes':
            complication_risks = self.assess_complication_risks()
        else:
            complication_risks = {}
        
        return {
            'risk_assessment': risk_assessment,
            'recommendations': recommendations,
            'complication_risks': complication_risks,
            'explanation': self.explanation
        }


# Helper function to convert assessment form data to patient data for expert system
def prepare_patient_data(assessment_data):
    """
    Convert assessment form data to a format the expert system can use.
    Handles both dictionary data and DiabetesAssessment objects.
    """
    patient_data = {}
    
    # Check if assessment_data is a Django model instance or a dictionary
    is_model = not isinstance(assessment_data, dict)
    
    # Basic demographics
    if is_model:
        # Handle DiabetesAssessment object
        if hasattr(assessment_data, 'age') and assessment_data.age:
            patient_data['age'] = assessment_data.age
        
        if hasattr(assessment_data, 'gender') and assessment_data.gender:
            patient_data['gender'] = assessment_data.gender
        
        # Calculate BMI if height and weight are available
        if hasattr(assessment_data, 'height') and hasattr(assessment_data, 'weight') and assessment_data.height and assessment_data.weight:
            height_m = assessment_data.height / 100  # convert cm to meters
            weight_kg = assessment_data.weight
            patient_data['weight'] = weight_kg
            patient_data['height'] = assessment_data.height
            patient_data['bmi'] = weight_kg / (height_m ** 2)
        elif hasattr(assessment_data, 'bmi') and assessment_data.bmi:
            # Use pre-calculated BMI if available
            patient_data['bmi'] = assessment_data.bmi
    else:
        # Handle dictionary data
        if 'age' in assessment_data and assessment_data['age']:
            patient_data['age'] = assessment_data['age']
        
        if 'gender' in assessment_data and assessment_data['gender']:
            patient_data['gender'] = assessment_data['gender']
        
        # Calculate BMI if height and weight are available
        if assessment_data.get('height') and assessment_data.get('weight'):
            height_m = assessment_data['height'] / 100  # convert cm to meters
            weight_kg = assessment_data['weight']
            patient_data['weight'] = weight_kg
            patient_data['height'] = assessment_data['height']
            patient_data['bmi'] = weight_kg / (height_m ** 2)
        elif 'bmi' in assessment_data and assessment_data['bmi']:
            patient_data['bmi'] = assessment_data['bmi']
    
    # Risk factors
    if is_model:
        # Handle model attributes
        if hasattr(assessment_data, 'family_history'):
            patient_data['family_history'] = assessment_data.family_history
        
        if hasattr(assessment_data, 'hypertension'):
            patient_data['hypertension'] = assessment_data.hypertension
        
        if hasattr(assessment_data, 'heart_disease'):
            patient_data['heart_disease'] = assessment_data.heart_disease
        
        if hasattr(assessment_data, 'physical_activity'):
            patient_data['physical_activity'] = assessment_data.physical_activity
        
        # Handle different formats of smoking data
        if hasattr(assessment_data, 'smoking_history') and assessment_data.smoking_history:
            smoking_status = assessment_data.smoking_history
            patient_data['smoking_history'] = smoking_status
            # Set smoking boolean based on smoking history
            if smoking_status in ['current', 'former', 'ever']:
                patient_data['smoking'] = True
            else:
                patient_data['smoking'] = False
        elif hasattr(assessment_data, 'smoking'):
            patient_data['smoking'] = assessment_data.smoking
    else:
        # Handle dictionary keys
        if 'family_history' in assessment_data:
            patient_data['family_history'] = assessment_data['family_history']
        
        if 'hypertension' in assessment_data:
            patient_data['hypertension'] = assessment_data['hypertension']
        
        if 'heart_disease' in assessment_data:
            patient_data['heart_disease'] = assessment_data['heart_disease']
        
        if 'physical_activity' in assessment_data:
            patient_data['physical_activity'] = assessment_data['physical_activity']
        
        # Handle different formats of smoking data
        if 'smoking_history' in assessment_data and assessment_data['smoking_history']:
            smoking_status = assessment_data['smoking_history']
            patient_data['smoking_history'] = smoking_status
            # Set smoking boolean based on smoking history
            if smoking_status in ['current', 'former', 'ever']:
                patient_data['smoking'] = True
            else:
                patient_data['smoking'] = False
        elif 'smoking' in assessment_data:
            patient_data['smoking'] = assessment_data['smoking']
    
    # Symptoms - both classic diabetes symptoms and our new ones
    if is_model:
        for symptom in kb.SYMPTOMS:
            if hasattr(assessment_data, symptom):
                patient_data[symptom] = getattr(assessment_data, symptom)
        
        # Add our new symptoms if they're not in the standard KB
        if hasattr(assessment_data, 'skin_darkening'):
            patient_data['skin_darkening'] = assessment_data.skin_darkening
        
        if hasattr(assessment_data, 'frequent_infections'):
            patient_data['frequent_infections'] = assessment_data.frequent_infections
        
        # Clinical measurements
        # Glucose measurements
        if hasattr(assessment_data, 'fasting_glucose') and assessment_data.fasting_glucose:
            patient_data['fasting_glucose'] = assessment_data.fasting_glucose
        
        if hasattr(assessment_data, 'hba1c') and assessment_data.hba1c:
            patient_data['hba1c'] = assessment_data.hba1c
        
        # Random glucose field removed as it's not in the NHANES dataset
        
        # Blood pressure
        if hasattr(assessment_data, 'blood_pressure_systolic') and assessment_data.blood_pressure_systolic:
            patient_data['systolic_bp'] = assessment_data.blood_pressure_systolic
        
        if hasattr(assessment_data, 'blood_pressure_diastolic') and assessment_data.blood_pressure_diastolic:
            patient_data['diastolic_bp'] = assessment_data.blood_pressure_diastolic
        
        # Lipid profile
        if hasattr(assessment_data, 'cholesterol') and assessment_data.cholesterol:
            patient_data['total_cholesterol'] = assessment_data.cholesterol
            
        if hasattr(assessment_data, 'hdl_cholesterol') and assessment_data.hdl_cholesterol:
            patient_data['hdl_cholesterol'] = assessment_data.hdl_cholesterol
            
        if hasattr(assessment_data, 'ldl_cholesterol') and assessment_data.ldl_cholesterol:
            patient_data['ldl_cholesterol'] = assessment_data.ldl_cholesterol
            
        if hasattr(assessment_data, 'triglycerides') and assessment_data.triglycerides:
            patient_data['triglycerides'] = assessment_data.triglycerides
    else:
        # Dictionary data
        for symptom in kb.SYMPTOMS:
            if symptom in assessment_data:
                patient_data[symptom] = assessment_data[symptom]
        
        # Add our new symptoms if they're not in the standard KB
        if 'skin_darkening' in assessment_data:
            patient_data['skin_darkening'] = assessment_data['skin_darkening']
        
        if 'frequent_infections' in assessment_data:
            patient_data['frequent_infections'] = assessment_data['frequent_infections']
        
        # Clinical measurements
        # Glucose measurements
        if 'fasting_glucose' in assessment_data and assessment_data['fasting_glucose']:
            patient_data['fasting_glucose'] = assessment_data['fasting_glucose']
        
        if 'hba1c' in assessment_data and assessment_data['hba1c']:
            patient_data['hba1c'] = assessment_data['hba1c']
        
        # Random glucose field removed as it's not in the NHANES dataset
        
        # Blood pressure
        if 'blood_pressure_systolic' in assessment_data and assessment_data['blood_pressure_systolic']:
            patient_data['systolic_bp'] = assessment_data['blood_pressure_systolic']
        
        if 'blood_pressure_diastolic' in assessment_data and assessment_data['blood_pressure_diastolic']:
            patient_data['diastolic_bp'] = assessment_data['blood_pressure_diastolic']
        
        # Lipid profile
        if 'cholesterol' in assessment_data and assessment_data['cholesterol']:
            patient_data['total_cholesterol'] = assessment_data['cholesterol']
            
        if 'hdl_cholesterol' in assessment_data and assessment_data['hdl_cholesterol']:
            patient_data['hdl_cholesterol'] = assessment_data['hdl_cholesterol']
            
        if 'ldl_cholesterol' in assessment_data and assessment_data['ldl_cholesterol']:
            patient_data['ldl_cholesterol'] = assessment_data['ldl_cholesterol']
            
        if 'triglycerides' in assessment_data and assessment_data['triglycerides']:
            patient_data['triglycerides'] = assessment_data['triglycerides']
    
    # Complication risk factors and warning signs
    for complication, data in kb.COMPLICATIONS.items():
        for factor in data['risk_factors']:
            if is_model:
                if hasattr(assessment_data, factor):
                    patient_data[factor] = getattr(assessment_data, factor)
            else:
                if factor in assessment_data:
                    patient_data[factor] = assessment_data[factor]
        
        for sign in data['warning_signs']:
            if is_model:
                if hasattr(assessment_data, sign):
                    patient_data[sign] = getattr(assessment_data, sign)
            else:
                if sign in assessment_data:
                    patient_data[sign] = assessment_data[sign]
    
    return patient_data
