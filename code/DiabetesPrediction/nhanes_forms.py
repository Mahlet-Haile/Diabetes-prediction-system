"""
Forms specifically designed for NHANES-based diabetes assessment and prediction
"""
from django import forms
from .models import DiabetesAssessment, Patient

class NHANESAssessmentForm(forms.ModelForm):
    """Form for diabetes assessment using NHANES dataset structure"""
    
    # Field for selecting an existing patient
    patient_selection = forms.ModelChoiceField(
        queryset=None, 
        required=False, 
        empty_label="Select Existing Patient", 
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # Field for entering a new patient's name
    new_patient_name = forms.CharField(
        max_length=255, 
        required=False, 
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter New Patient Name'})
    )
    
    class Meta:
        model = DiabetesAssessment
        fields = [
            # Patient information
            'patient_selection', 'new_patient_name',
            
            # Demographics
            'gender', 'age',
            
            # Anthropometrics
            'height', 'weight',
            
            # Clinical measurements
            'blood_pressure_systolic', 'blood_pressure_diastolic',
            'fasting_glucose', 'hba1c', # 'glucose' field removed as it's not in the NHANES dataset
            'cholesterol',
            
            # Risk factors
            'smoking_history', 'hypertension', 'heart_disease',
            
            # Symptoms
            'polyuria', 'polydipsia', 'polyphagia', 'weight_loss',
            'fatigue', 'blurred_vision', 'slow_healing', 'tingling'
        ]
        
        # Define widgets for better form rendering
        widgets = {
            # Demographics
            'gender': forms.Select(attrs={'class': 'form-control'}),
            'age': forms.NumberInput(attrs={'class': 'form-control', 'min': '18', 'max': '85', 'placeholder': 'years', 'oninvalid': "this.setCustomValidity('Please enter an age between 18 and 85 years')", 'oninput': "this.setCustomValidity('')"}),
            
            # Anthropometrics
            'height': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '120', 'max': '250', 'placeholder': 'cm', 'oninvalid': "this.setCustomValidity('Please enter a height between 120 and 250 cm')", 'oninput': "this.setCustomValidity('')"}),
            'weight': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '30', 'max': '300', 'placeholder': 'kg', 'oninvalid': "this.setCustomValidity('Please enter a weight between 30 and 300 kg')", 'oninput': "this.setCustomValidity('')"}),
            
            # Clinical measurements
            'blood_pressure_systolic': forms.NumberInput(attrs={'class': 'form-control', 'min': '70', 'max': '250', 'placeholder': 'mmHg'}),
            'blood_pressure_diastolic': forms.NumberInput(attrs={'class': 'form-control', 'min': '40', 'max': '150', 'placeholder': 'mmHg'}),
            # 'glucose' field removed as it's not in the NHANES dataset
            'fasting_glucose': forms.NumberInput(attrs={'class': 'form-control', 'step': '1', 'min': '50', 'max': '500', 'placeholder': 'mg/dL'}),
            'hba1c': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '4', 'max': '14', 'placeholder': '%'}),
            'cholesterol': forms.NumberInput(attrs={'class': 'form-control', 'step': '1', 'min': '100', 'max': '400', 'placeholder': 'mg/dL'}),
            
            # Risk factors
            'smoking_history': forms.Select(attrs={'class': 'form-control'}),
            'hypertension': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'heart_disease': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            
            # Symptoms
            'polyuria': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'polydipsia': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'polyphagia': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'weight_loss': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'fatigue': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'blurred_vision': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'slow_healing': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'tingling': forms.CheckboxInput(attrs={'class': 'form-check-input'})
        }
    
    def __init__(self, *args, **kwargs):
        # Get the doctor user from kwargs to filter patients
        self.doctor = kwargs.pop('doctor', None)
        super(NHANESAssessmentForm, self).__init__(*args, **kwargs)
        
        # Set the queryset for patient selection based on the doctor
        if self.doctor:
            self.fields['patient_selection'].queryset = Patient.objects.filter(doctor=self.doctor)
        else:
            self.fields['patient_selection'].queryset = Patient.objects.none()
        
        # Add help text and labels for NHANES fields
        # Demographics
        self.fields['gender'].label = "Gender"
        self.fields['age'].label = "Age"
        
        # Anthropometrics
        self.fields['height'].help_text = "Height in centimeters"
        self.fields['weight'].help_text = "Weight in kilograms"
        
        # Clinical measurements
        self.fields['blood_pressure_systolic'].label = "Systolic Blood Pressure"
        self.fields['blood_pressure_systolic'].help_text = "Upper number in blood pressure reading (mmHg)"
        self.fields['blood_pressure_diastolic'].label = "Diastolic Blood Pressure"
        self.fields['blood_pressure_diastolic'].help_text = "Lower number in blood pressure reading (mmHg)"
        self.fields['cholesterol'].label = "Total Cholesterol Level"
        # 'glucose' field removed as it's not in the NHANES dataset
        self.fields['fasting_glucose'].label = "Fasting Glucose"
        self.fields['fasting_glucose'].help_text = "Blood glucose after at least 8 hours of fasting (mg/dL)"
        self.fields['hba1c'].label = "HbA1c"
        self.fields['hba1c'].help_text = "Glycated hemoglobin percentage (%)"
        
        # Risk factors
        self.fields['smoking_history'].label = "Smoking History"
        self.fields['hypertension'].label = "Hypertension"
        self.fields['hypertension'].help_text = "Diagnosed with high blood pressure"
        self.fields['heart_disease'].label = "Heart Disease"
        self.fields['heart_disease'].help_text = "Diagnosed with any cardiovascular condition"
        
        # Symptoms
        self.fields['polyuria'].label = "Frequent Urination"
        self.fields['polydipsia'].label = "Excessive Thirst"
        self.fields['polyphagia'].label = "Excessive Hunger"
        self.fields['weight_loss'].label = "Unexplained Weight Loss"
        self.fields['fatigue'].label = "Unusual Fatigue"
        self.fields['blurred_vision'].label = "Blurred Vision"
        self.fields['slow_healing'].label = "Slow Healing of Cuts/Wounds"
        self.fields['tingling'].label = "Numbness or Tingling in Hands/Feet"
        
        # Make all clinical measurements optional
        self.fields['height'].required = False
        self.fields['weight'].required = False
        self.fields['blood_pressure_systolic'].required = False
        self.fields['blood_pressure_diastolic'].required = False
        self.fields['cholesterol'].required = False
        # 'glucose' field removed as it's not in the NHANES dataset
        self.fields['fasting_glucose'].required = False
        self.fields['hba1c'].required = False
