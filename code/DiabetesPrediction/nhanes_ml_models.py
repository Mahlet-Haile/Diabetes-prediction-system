"""NHANES Diabetes Prediction Models

ML models for diabetes prediction using NHANES dataset.
This module extends the existing ML pipeline to work with NHANES data structure.
This module provides a simplified interface to load and use the trained NHANES diabetes prediction models.
"""

import pandas as pd
import numpy as np
import os
import joblib
import logging

logger = logging.getLogger(__name__)

class NHANESDiabetesPredictionModel:
    """Machine learning model for diabetes prediction using NHANES data"""

    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
        os.makedirs(self.model_path, exist_ok=True)
        self.diabetes_model_path = os.path.join(self.model_path, 'diabetes_model_nhanes.joblib')
        self.type_model_path = os.path.join(self.model_path, 'diabetes_type_model_nhanes.joblib')
        
        # Initialize models
        self.diabetes_model = None
        self.type_model = None
        
        # Define NHANES feature groups
        # Demographics and anthropometrics
        self.demographic_features = ['age', 'gender', 'height', 'weight', 'bmi']
        
        # Clinical measurements
        self.clinical_features = ['blood_pressure_systolic', 'blood_pressure_diastolic',
                                 'fasting_glucose', 'hba1c', 
                                 'cholesterol', 'hdl', 'ldl', 'triglycerides',
                                 'creatinine', 'egfr', 'urine_albumin', 'urine_creatinine',
                                 'insulin', 'c_peptide', 'homa_ir']
        # Note: 'glucose' (random glucose) is excluded as it's not in the NHANES dataset
        # It is only used in the rule-based expert system
        
        # Risk factors
        self.risk_features = ['smoking_history', 'smoking', 'alcohol', 'active',
                             'physical_activity', 'family_history', 'hypertension',
                             'heart_disease']
        
        # Symptoms
        self.symptom_features = ['polyuria', 'polydipsia', 'polyphagia', 'weight_loss',
                                'fatigue', 'blurred_vision', 'slow_healing', 'tingling']
        
        # Complications
        self.complication_features = ['chest_pain', 'shortness_of_breath', 'swelling_in_legs',
                                     'numbness', 'foot_ulcers', 'vision_loss']
        
        # Medical history
        self.history_features = ['gestational_diabetes', 'autoimmune_disease', 'pcos']
        
        # All features combined
        self.all_features = (self.demographic_features + self.clinical_features + 
                            self.risk_features + self.symptom_features + 
                            self.complication_features + self.history_features)
        
        # Target variables
        self.target_variables = ['diabetes', 'prediabetes', 'diabetes_type']
        
        # Load models if they exist
        self.load_models()
        
    def load_models(self):
        """Load trained models if they exist"""
        try:
            if os.path.exists(self.diabetes_model_path):
                self.diabetes_model = joblib.load(self.diabetes_model_path)
                
            if os.path.exists(self.type_model_path):
                self.type_model = joblib.load(self.type_model_path)
                
            # Add a method to check if models exist
            return True
        except Exception as e:
            # Log error instead of print
            import logging
            logging.error(f"Error loading NHANES models: {e}")
            return False
            
    def models_exist(self):
        """Check if the NHANES models exist"""
        return (os.path.exists(self.diabetes_model_path) and 
                os.path.exists(self.type_model_path) and
                self.diabetes_model is not None and
                self.type_model is not None)
    
    def preprocess_data(self, data):
        """
        Preprocess data for prediction using NHANES feature structure
        
        Args:
            data: Dictionary or DataFrame with patient data
            
        Returns:
            Processed DataFrame ready for model prediction
        """
        # Convert dictionary to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Create a processed DataFrame with expected features
        processed_df = pd.DataFrame()
        
        # Map gender if present (accept various formats)
        if 'gender' in df.columns:
            gender_mapping = {
                'M': 1, 'm': 1, 'male': 1, 'Male': 1, 1: 1,
                'F': 2, 'f': 2, 'female': 2, 'Female': 2, 2: 2
            }
            processed_df['gender'] = df['gender'].map(lambda x: gender_mapping.get(x, 1))
        else:
            processed_df['gender'] = 1  # Default to male if not specified
        
        # Process all numeric features
        numeric_features = [f for f in self.all_features if f != 'gender' and f != 'smoking_history']
        for feature in numeric_features:
            if feature in df.columns:
                processed_df[feature] = df[feature]
            else:
                # Use reasonable defaults for missing features
                if feature in ['age', 'height', 'weight', 'bmi']:
                    defaults = {'age': 50, 'height': 170, 'weight': 70, 'bmi': 24}
                    processed_df[feature] = defaults.get(feature, 0)
                elif feature in self.clinical_features:
                    # Default to normal values for clinical measurements
                    defaults = {
                        'blood_pressure_systolic': 120, 'blood_pressure_diastolic': 80,
                        'fasting_glucose': 90, 'hba1c': 5.5,
                        'cholesterol': 180, 'hdl': 50, 'ldl': 100, 'triglycerides': 150,
                        'creatinine': 0.9, 'egfr': 90, 'urine_albumin': 5, 'urine_creatinine': 100,
                        'insulin': 10, 'c_peptide': 2, 'homa_ir': 2.5
                    }
                    # Note: 'glucose' (random glucose) is excluded as it's not in the NHANES dataset
                    processed_df[feature] = defaults.get(feature, 0)
                else:
                    # Default to 0 for binary features (risk factors, symptoms, etc.)
                    processed_df[feature] = 0
        
        # Process smoking history
        if 'smoking_history' in df.columns:
            smoking_options = ['never', 'former', 'current', 'ever', 'no_info']
            if df['smoking_history'].iloc[0] in smoking_options:
                processed_df['smoking_history'] = df['smoking_history']
            else:
                processed_df['smoking_history'] = 'no_info'
        else:
            processed_df['smoking_history'] = 'no_info'
        
        # Create derived features if needed
        if 'blood_glucose_level' not in processed_df.columns:
            if 'fasting_glucose' in processed_df.columns:
                processed_df['blood_glucose_level'] = processed_df['fasting_glucose']
            else:
                processed_df['blood_glucose_level'] = 90  # Default normal value
            # Note: 'glucose' (random glucose) option removed as it's not in the NHANES dataset
        
        if 'hba1c_level' not in processed_df.columns and 'hba1c' in processed_df.columns:
            processed_df['hba1c_level'] = processed_df['hba1c']
        elif 'hba1c_level' not in processed_df.columns:
            processed_df['hba1c_level'] = 5.5  # Default normal value
        
        # Map symptoms to the original model's expected feature names
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
            if nhanes_name in processed_df.columns:
                processed_df[model_name] = processed_df[nhanes_name]
            else:
                processed_df[model_name] = 0
        
        # Encode categorical variables for the model
        if 'gender' in processed_df.columns:
            processed_df['gender_encoded'] = processed_df['gender'].map({1: 0, 2: 1})
        else:
            processed_df['gender_encoded'] = 0
        
        if 'smoking_history' in processed_df.columns:
            # Simplified smoking map that directly maps to binary values
            smoking_map = {
                'not_smoker': 0,  # Not a smoker -> 0 
                'smoker': 1,      # Smoker -> 1
                'no_info': 0       # Default to not smoker
            }
            processed_df['smoking_encoded'] = processed_df['smoking_history'].map(smoking_map)
            # If smoking_history is already binary (0/1), use it directly
            if processed_df['smoking_history'].dtype == 'int64' or processed_df['smoking_history'].dtype == 'float64':
                processed_df['smoking_encoded'] = processed_df['smoking_history']
        else:
            processed_df['smoking_encoded'] = 0  # Default to not smoker
        
        return processed_df
    
    def predict(self, data):
        """
        Predict diabetes risk and type from input data
        
        Args:
            data: Dictionary or DataFrame with patient data
            
        Returns:
            Dictionary with prediction results
        """
        if self.diabetes_model is None or self.type_model is None:
            print("Models not loaded. Please train the models first.")
            return None
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Make predictions
        try:
            # Diabetes detection
            diabetes_prob = self.diabetes_model.predict_proba(processed_data)[:, 1]
            diabetes_prediction = (diabetes_prob > 0.5).astype(int)
            
            # Diabetes type prediction
            diabetes_type = self.type_model.predict(processed_data)
            type_probs = self.type_model.predict_proba(processed_data)
            
            # Get the confidence score (probability of the predicted class)
            confidence_scores = []
            diabetes_types = self.type_model.classes_
            
            for i, type_pred in enumerate(diabetes_type):
                type_idx = list(diabetes_types).index(type_pred)
                confidence_scores.append(type_probs[i, type_idx])
            
            # Calculate risk score (0-100)
            # For those predicted as diabetic, base on probability
            # For those predicted as prediabetic, scale from 30-70
            # For those predicted as normal, scale from 0-30
            risk_scores = []
            
            for i, (d_pred, d_type) in enumerate(zip(diabetes_prediction, diabetes_type)):
                if d_pred == 1:  # Diabetic
                    risk_scores.append(70 + (diabetes_prob[i] * 30))
                elif d_type == 'prediabetes':  # Prediabetic
                    prediabetes_prob = type_probs[i, list(diabetes_types).index('prediabetes')]
                    risk_scores.append(30 + (prediabetes_prob * 40))
                else:  # Normal
                    risk_scores.append(diabetes_prob[i] * 30)
            
            # Prepare results
            results = []
            for i in range(len(processed_data)):
                result = {
                    'diabetes': bool(diabetes_prediction[i]),
                    'prediction_probability': float(diabetes_prob[i]),
                    'diabetes_type': diabetes_type[i],
                    'confidence_score': float(confidence_scores[i]),
                    'risk_score': float(risk_scores[i])
                }
                
                # Add diagnoses and recommendations
                result['diagnosis'] = self.generate_diagnosis(
                    diabetes_type[i], 
                    result['risk_score'],
                    processed_data.iloc[i]
                )
                
                result['recommendations'] = {
                    'diet': self.generate_diet_recommendations(diabetes_type[i], processed_data.iloc[i:i+1]),
                    'exercise': self.generate_exercise_recommendations(diabetes_type[i], processed_data.iloc[i:i+1]),
                    'monitoring': self.generate_monitoring_recommendations(diabetes_type[i], processed_data.iloc[i:i+1])
                }
                
                results.append(result)
            
            return results[0] if len(results) == 1 else results
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_diagnosis(self, diabetes_type, risk_score, data):
        """Generate diagnosis text based on diabetes type and risk score"""
        if diabetes_type == 'type1':
            return "Type 1 Diabetes Mellitus: An autoimmune condition where the body's immune system destroys the insulin-producing beta cells in the pancreas."
        
        elif diabetes_type == 'type2':
            if risk_score >= 90:
                return "Severe Type 2 Diabetes Mellitus: A chronic condition that affects the way your body metabolizes sugar. Very high risk profile with likely complications."
            elif risk_score >= 80:
                return "High-risk Type 2 Diabetes Mellitus: Significant insulin resistance and/or inadequate insulin production. Requires immediate intervention."
            else:
                return "Type 2 Diabetes Mellitus: A chronic condition characterized by high blood sugar levels due to insulin resistance and inadequate insulin production."
        
        elif diabetes_type == 'prediabetes':
            if risk_score >= 60:
                return "High-risk Prediabetes: Blood glucose levels are elevated but not yet in the diabetic range. High risk of progression to Type 2 diabetes within 5 years."
            else:
                return "Prediabetes: Blood glucose levels are higher than normal but not yet high enough to be diagnosed as diabetes. Lifestyle modifications strongly recommended."
        
        else:  # No diabetes
            if risk_score >= 20:
                return "Elevated Diabetes Risk: While not currently diabetic, your risk factors suggest an increased likelihood of developing diabetes in the future."
            else:
                return "Normal Glucose Metabolism: No evidence of diabetes or prediabetes at this time."
    
    def generate_diet_recommendations(self, diabetes_type, data):
        """Generate diet recommendations based on diabetes type and patient data"""
        bmi = data['bmi'].values[0] if 'bmi' in data.columns else 25
        
        if diabetes_type == 'none':
            if bmi > 25:
                return ("Focus on a balanced diet with plenty of fruits, vegetables, whole grains, and lean proteins. "
                        "Consider portion control to achieve and maintain a healthy weight.")
            else:
                return ("Maintain a balanced diet with a variety of nutrients. Include fruits, vegetables, "
                        "whole grains, lean proteins, and healthy fats.")
        
        if diabetes_type == 'prediabetes':
            if bmi > 30:
                return ("Focus on weight loss through a reduced-calorie diet. Choose foods with low glycemic index. "
                        "Limit refined carbohydrates, sugary beverages, and processed foods. "
                        "Aim for regular meal times and consistent carbohydrate intake.")
            else:
                return ("Choose foods with low glycemic index. Limit refined carbohydrates, sugary beverages, "
                        "and processed foods. Emphasize vegetables, whole grains, lean proteins, and healthy fats. "
                        "Maintain consistent meal timing.")
        
        if diabetes_type == 'type1':
            return ("Consistency in carbohydrate intake is key to managing blood glucose. "
                    "Learn carbohydrate counting to match insulin doses with food intake. "
                    "Eat regular meals and snacks to avoid blood sugar fluctuations. "
                    "Work with a registered dietitian for personalized guidance.")
        
        if diabetes_type == 'type2':
            if bmi > 30:
                return ("Focus on weight loss through calorie reduction and portion control. "
                        "Choose high-fiber, low-glycemic foods. Limit carbohydrates, especially refined grains and sugars. "
                        "Include lean proteins, healthy fats, and plenty of non-starchy vegetables. "
                        "Maintain consistent meal timing to help regulate blood glucose.")
            else:
                return ("Choose high-fiber, low-glycemic foods that release glucose slowly. "
                        "Limit refined carbohydrates and added sugars. Include lean proteins and healthy fats at each meal. "
                        "Maintain consistent meal timing and portion control.")
        
        return "Consult with a healthcare provider for personalized dietary recommendations."
    
    def generate_exercise_recommendations(self, diabetes_type, data):
        """Generate exercise recommendations based on diabetes type and patient data"""
        age = data['age'].values[0] if 'age' in data.columns else 50
        has_heart_disease = data['heart_disease'].values[0] if 'heart_disease' in data.columns else 0
        
        if has_heart_disease:
            return ("Consult with your healthcare provider before starting any exercise program. "
                    "Low-impact activities like walking, swimming, or stationary cycling may be appropriate "
                    "with proper medical clearance and supervision.")
        
        if diabetes_type == 'none':
            return ("Aim for at least 150 minutes of moderate-intensity aerobic activity per week, "
                    "plus muscle-strengthening activities twice per week.")
        
        if diabetes_type == 'prediabetes':
            return ("Aim for 150-300 minutes of moderate-intensity exercise per week. "
                    "Include both aerobic activities and strength training. "
                    "Even short activity sessions can help regulate blood sugar levels.")
        
        if diabetes_type == 'type1':
            return ("Regular exercise is important, but requires careful blood sugar monitoring before, during, and after activity. "
                    "Carry fast-acting carbohydrates during exercise. Consider adjusting insulin doses before planned exercise. "
                    "Aim for 150 minutes of moderate activity per week with strength training twice weekly.")
        
        if diabetes_type == 'type2':
            if age > 65:
                return ("Aim for 150 minutes of moderate activity weekly, such as walking, swimming, or cycling. "
                        "Add gentle strength training and flexibility exercises. "
                        "Monitor blood sugar before and after exercise. Start slowly and gradually increase intensity.")
            else:
                return ("Aim for 150-300 minutes of moderate exercise weekly. "
                        "Include both cardio and strength training. Exercise can improve insulin sensitivity for up to 24-48 hours. "
                        "Monitor blood glucose before and after exercise, especially if taking insulin or medications.")
        
        return "Consult with a healthcare provider for personalized exercise recommendations."
    
    def generate_monitoring_recommendations(self, diabetes_type, data):
        """Generate monitoring recommendations based on diabetes type and patient data"""
        if diabetes_type == 'none':
            return ("Get regular check-ups with your healthcare provider. "
                    "Consider annual screening for diabetes if you have risk factors.")
        
        if diabetes_type == 'prediabetes':
            return ("Check blood glucose levels at least annually. "
                    "Monitor for symptoms of diabetes. "
                    "Follow up with your healthcare provider every 3-6 months.")
        
        if diabetes_type == 'type1':
            return ("Check blood glucose 4-10 times daily, including before meals and at bedtime. "
                    "Get HbA1c tested every 3 months. "
                    "See an endocrinologist regularly. "
                    "Annual kidney, eye, and foot exams are essential.")
        
        if diabetes_type == 'type2':
            return ("Check blood glucose as recommended by your healthcare provider. "
                    "Get HbA1c tested every 3-6 months. "
                    "Regular blood pressure and cholesterol monitoring. "
                    "Annual kidney, eye, and foot exams recommended.")
        
        return "Consult with a healthcare provider for personalized monitoring recommendations."
    
    def train_models(self, dataset_path):
        """Train the diabetes prediction models using NHANES dataset"""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Load the NHANES dataset
            df = pd.read_csv(dataset_path)
            
            # Display basic stats
            record_count = len(df)
            diabetes_count = df['diabetes'].sum()
            prediabetes_count = df['prediabetes'].sum()
            normal_count = record_count - diabetes_count - prediabetes_count
            
            logger.info(f"Training NHANES models with {record_count} records:")
            logger.info(f"  - {diabetes_count} ({diabetes_count/record_count*100:.1f}%) diabetes cases")
            logger.info(f"  - {prediabetes_count} ({prediabetes_count/record_count*100:.1f}%) prediabetes cases")
            logger.info(f"  - {normal_count} ({normal_count/record_count*100:.1f}%) non-diabetic cases")
            
            # Create diabetes_type column if it doesn't exist
            if 'diabetes_type' not in df.columns:
                logger.info("Creating diabetes_type column from diabetes and prediabetes flags")
                # Initialize as 'none' for everyone
                df['diabetes_type'] = 'none'
                
                # Set prediabetes cases
                df.loc[df['prediabetes'] == 1, 'diabetes_type'] = 'prediabetes'
                
                # Set diabetes cases (assuming type 2 for most, and a small percentage as type 1)
                df.loc[df['diabetes'] == 1, 'diabetes_type'] = 'type2'
                
                # Randomly assign 10% of diabetes cases as type 1
                type1_indices = df[df['diabetes'] == 1].sample(frac=0.1, random_state=42).index
                df.loc[type1_indices, 'diabetes_type'] = 'type1'
                
                # Verify the distribution
                type_counts = df['diabetes_type'].value_counts()
                logger.info(f"Diabetes type distribution: {type_counts.to_dict()}")
            
            # Process categorical features
            # Gender encoding (1=Male, 2=Female)
            print(f"DEBUG: Gender values in dataset: {df['gender'].unique()}")
            
            # Check if gender column exists and handle possible errors
            if 'gender' in df.columns:
                try:
                    # Convert gender to numeric if it's not already
                    if df['gender'].dtype == 'object':
                        df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
                    
                    # Map gender values safely with a default
                    df['gender_encoded'] = df['gender'].map({1: 0, 2: 1}).fillna(0)
                except Exception as e:
                    print(f"DEBUG: Error encoding gender: {str(e)}")
                    # Create a default encoding as fallback
                    df['gender_encoded'] = 0
            else:
                print("DEBUG: No gender column found in dataset")
                df['gender_encoded'] = 0  # Default value
            
            # Smoking history encoding with error handling
            smoking_map = {
                'never': 0, 
                'former': 1, 
                'current': 2, 
                'ever': 3, 
                'no_info': 4
            }
            
            # Smoking history encoding with error handling
            smoking_map = {
                'never': 0, 
                'former': 1, 
                'current': 2, 
                'ever': 3, 
                'no_info': 4
            }
            
            try:
                if 'smoking_history' in df.columns:
                    print(f"DEBUG: smoking_history values: {df['smoking_history'].unique()}")
                    df['smoking_encoded'] = df['smoking_history'].map(smoking_map).fillna(4)  # Default to no_info
                elif 'smoking' in df.columns:
                    # Use binary smoking indicator instead
                    print("DEBUG: Using binary 'smoking' column instead of 'smoking_history'")
                    df['smoking_encoded'] = df['smoking'].fillna(0).astype(int)
                else:
                    print("DEBUG: No smoking_history or smoking column found, using default")
                    df['smoking_encoded'] = 4  # Default to no_info
            except Exception as e:
                print(f"DEBUG: Error encoding smoking: {str(e)}")
                df['smoking_encoded'] = 4  # Default to no_info
            # Process symptoms for features list
            symptom_features = ['polyuria', 'polydipsia', 'polyphagia', 'weight_loss',
                              'fatigue', 'blurred_vision', 'slow_healing', 'tingling']
            
            for symptom in symptom_features:
                if symptom in df.columns:
                    features.append(symptom)
            
            # Create train/test split
            X = df[features]
            y_diabetes = df['diabetes']  # Binary: has diabetes or not
            
            # For type classification, we use records with diabetes OR prediabetes to ensure multiple classes
            diabetes_df = df[(df['diabetes'] == 1) | (df['prediabetes'] == 1)].copy()
            
            # Make sure we have at least 2 classes by verifying the type distribution
            type_count = diabetes_df['diabetes_type'].value_counts()
            logger.info(f"Type classification distribution: {type_count.to_dict()}")
            
            # Ensure we have at least 2 samples of each class to avoid split issues
            min_samples_per_class = 2
            for dtype, count in type_count.items():
                if count < min_samples_per_class:
                    # Add duplicated samples to reach minimum count
                    samples_to_add = min_samples_per_class - count
                    extra_samples = diabetes_df[diabetes_df['diabetes_type'] == dtype].sample(n=1, replace=True)
                    # Duplicate the samples as needed
                    extra_samples = pd.concat([extra_samples] * samples_to_add, ignore_index=True)
                    diabetes_df = pd.concat([diabetes_df, extra_samples], ignore_index=True)
            
            X_type = diabetes_df[features]
            y_type = diabetes_df['diabetes_type']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y_diabetes, test_size=0.2, random_state=42)
            
            # Train diabetes detection model (Random Forest)
            logger.info("Training diabetes detection model...")
            diabetes_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            diabetes_pipeline.fit(X_train, y_train)
            
            # Evaluate diabetes model
            y_pred = diabetes_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Diabetes detection accuracy: {accuracy:.4f}")
            
            # Train diabetes type model (Gradient Boosting Classifier)
            if len(diabetes_df) > 0:
                logger.info("Training diabetes type model...")
                X_type_train, X_type_test, y_type_train, y_type_test = train_test_split(
                    X_type, y_type, test_size=0.2, random_state=42)
                
                type_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
                ])
                
                type_pipeline.fit(X_type_train, y_type_train)
                
                # Evaluate type model
                y_type_pred = type_pipeline.predict(X_type_test)
                type_accuracy = accuracy_score(y_type_test, y_type_pred)
                logger.info(f"Diabetes type classification accuracy: {type_accuracy:.4f}")
            else:
                logger.warning("No diabetes cases found for training the type model. Using default model.")
                # Create a simple default model that always predicts type 2
                type_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', GradientBoostingClassifier(n_estimators=10, random_state=42))
                ])
                
                # Create a minimal training set with two classes
                X_minimal = X.iloc[:2].copy()
                y_minimal = pd.Series(['type1', 'type2'])
                type_pipeline.fit(X_minimal, y_minimal)
            
            # Save the models
            logger.info(f"Saving models to {self.model_path}")
            joblib.dump(diabetes_pipeline, self.diabetes_model_path)
            joblib.dump(type_pipeline, self.type_model_path)
            
            # Update the instance models
            self.diabetes_model = diabetes_pipeline
            self.type_model = type_pipeline
            
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
