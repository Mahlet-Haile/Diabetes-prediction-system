"""
Comprehensive assessment form integrating NHANES dataset parameters with expert system inputs
"""
from django import forms
from .models import DiabetesAssessment, Patient

class ComprehensiveDiabetesAssessmentForm(forms.ModelForm):
    """Comprehensive form for diabetes assessment combining NHANES data structure with expert system inputs"""
    
    # Patient selection fields
    patient_selection = forms.ModelChoiceField(
        queryset=None, 
        required=False, 
        empty_label="Select Existing Patient", 
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    new_patient_name = forms.CharField(
        max_length=255, 
        required=False, 
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter First Name'})
    )
    
    new_patient_last_name = forms.CharField(
        max_length=255, 
        required=False, 
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Last Name'})
    )
    
    # Explicitly define gender field to ensure proper form rendering and initialization
    gender = forms.ChoiceField(
        choices=DiabetesAssessment.GENDER_CHOICES,
        required=True,
        widget=forms.Select(attrs={'class': 'form-control', 'id': 'id_gender_field'})
    )
    
    # Additional fields for expert system integration
    hdl_cholesterol = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1',
            'min': '20',
            'max': '100',
            'placeholder': 'mg/dL'
        })
    )
    
    ldl_cholesterol = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1',
            'min': '40',
            'max': '300',
            'placeholder': 'mg/dL'
        })
    )
    
    triglycerides = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1',
            'min': '50',
            'max': '500',
            'placeholder': 'mg/dL'
        })
    )
    
    skin_darkening = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    frequent_infections = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    family_history = forms.BooleanField(
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'id_family_history_field',
            'style': 'margin-right: 5px;'
        })
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
            # 'glucose' field removed as it's not in the NHANES dataset
            'cholesterol', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 
            'fasting_glucose', 'hba1c',
            
            # Risk factors
            'smoking_history', 'hypertension', 'heart_disease', 'family_history',
            
            # Primary symptoms
            'polyuria', 'polydipsia', 'polyphagia', 'weight_loss',
            'fatigue', 'blurred_vision',
            
            # Secondary symptoms and complications
            'slow_healing', 'tingling', 
            'skin_darkening', 'frequent_infections'  # Added for expert system
        ]
        
        # Define widgets for better form rendering
        widgets = {
            # Demographics
            'gender': forms.Select(attrs={'class': 'form-control'}),
            'age': forms.NumberInput(attrs={'class': 'form-control', 'min': '18', 'max': '85', 'placeholder': 'years', 'oninvalid': "this.setCustomValidity('Please enter an age between 18 and 85 years')", 'oninput': "this.setCustomValidity('')"}),
            
            # Anthropometrics
            'height': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '120', 'max': '250', 'placeholder': 'cm', 'oninvalid': "this.setCustomValidity('Please enter a height between 120 and 250 cm')", 'oninput': "this.setCustomValidity('')"}),
            'weight': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '30', 'max': '300', 'placeholder': 'kg', 'oninvalid': "this.setCustomValidity('Please enter a weight between 30 and 300 kg')", 'oninput': "this.setCustomValidity('')"}),
            
            # Blood pressure
            'blood_pressure_systolic': forms.NumberInput(attrs={'class': 'form-control', 'min': '70', 'max': '250', 'placeholder': 'mmHg'}),
            'blood_pressure_diastolic': forms.NumberInput(attrs={'class': 'form-control', 'min': '40', 'max': '150', 'placeholder': 'mmHg'}),
            
            # Glucose measurements
            # 'glucose' field removed as it's not in the NHANES dataset
            'fasting_glucose': forms.NumberInput(attrs={'class': 'form-control', 'step': '1', 'min': '50', 'max': '500', 'placeholder': 'mg/dL'}),
            'hba1c': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '4', 'max': '14', 'placeholder': '%'}),
            
            # Lipid profile
            'cholesterol': forms.NumberInput(attrs={'class': 'form-control', 'step': '1', 'min': '100', 'max': '400', 'placeholder': 'mg/dL'}),
            'hdl_cholesterol': forms.NumberInput(attrs={'class': 'form-control', 'step': '1', 'min': '20', 'max': '100', 'placeholder': 'mg/dL'}),
            'ldl_cholesterol': forms.NumberInput(attrs={'class': 'form-control', 'step': '1', 'min': '40', 'max': '250', 'placeholder': 'mg/dL'}),
            'triglycerides': forms.NumberInput(attrs={'class': 'form-control', 'step': '1', 'min': '50', 'max': '1000', 'placeholder': 'mg/dL'}),
            
            # Risk factors
            'smoking_history': forms.Select(attrs={'class': 'form-control'}),
            'hypertension': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'heart_disease': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            
            # Primary symptoms
            'polyuria': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'polydipsia': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'polyphagia': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'weight_loss': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'fatigue': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'blurred_vision': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            
            # Secondary symptoms
            'slow_healing': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'tingling': forms.CheckboxInput(attrs={'class': 'form-check-input'})
        }
    
    def __init__(self, *args, **kwargs):
        # Get the doctor user from kwargs to filter patients
        self.doctor = kwargs.pop('doctor', None)
        super(ComprehensiveDiabetesAssessmentForm, self).__init__(*args, **kwargs)
        
        # Set the queryset for patient selection based on the doctor
        if self.doctor:
            self.fields['patient_selection'].queryset = Patient.objects.filter(doctor=self.doctor)
        else:
            self.fields['patient_selection'].queryset = Patient.objects.none()
        
        # Customize field labels and help text
        # Demographics
        self.fields['gender'].label = "Gender"
        self.fields['age'].label = "Age"
        
        # Anthropometrics
        self.fields['height'].label = "Height"
        self.fields['height'].help_text = "Height in centimeters (cm)"
        self.fields['weight'].label = "Weight"
        self.fields['weight'].help_text = "Weight in kilograms (kg)"
        
        # Blood pressure
        self.fields['blood_pressure_systolic'].label = "Systolic Blood Pressure"
        self.fields['blood_pressure_systolic'].help_text = "Upper number in blood pressure reading (mmHg)"
        self.fields['blood_pressure_diastolic'].label = "Diastolic Blood Pressure"
        self.fields['blood_pressure_diastolic'].help_text = "Lower number in blood pressure reading (mmHg)"
        
        # Glucose measurements
        # 'glucose' field removed as it's not in the NHANES dataset
        self.fields['fasting_glucose'].label = "Fasting Glucose"
        self.fields['fasting_glucose'].help_text = "Blood glucose after at least 8 hours of fasting (mg/dL)"
        self.fields['hba1c'].label = "HbA1c"
        self.fields['hba1c'].help_text = "Glycated hemoglobin percentage (%)"
        
        # Lipid profile
        self.fields['cholesterol'].label = "Total Cholesterol Level"
        self.fields['hdl_cholesterol'].label = "HDL Cholesterol"
        self.fields['hdl_cholesterol'].help_text = "High-density lipoprotein ('good' cholesterol) in mg/dL"
        self.fields['ldl_cholesterol'].label = "LDL Cholesterol"
        self.fields['ldl_cholesterol'].help_text = "Low-density lipoprotein ('bad' cholesterol) in mg/dL"
        self.fields['triglycerides'].label = "Triglycerides"
        self.fields['triglycerides'].help_text = "Blood fat level - strongly associated with diabetes and insulin resistance (mg/dL)"
        
        # Risk factors
        self.fields['smoking_history'].label = "Smoking History"
        self.fields['hypertension'].label = "High Blood Pressure (Hypertension)"
        self.fields['hypertension'].help_text = "Previously diagnosed with high blood pressure"
        self.fields['heart_disease'].label = "Cardiovascular Disease"
        self.fields['heart_disease'].help_text = "Previously diagnosed with any cardiovascular condition"
        self.fields['family_history'].label = "Family History of Diabetes"
        
        # Primary symptoms
        self.fields['polyuria'].label = "Frequent Urination"
        self.fields['polydipsia'].label = "Excessive Thirst"
        self.fields['polyphagia'].label = "Excessive Hunger"
        self.fields['weight_loss'].label = "Unexplained Weight Loss"
        self.fields['fatigue'].label = "Unusual Fatigue"
        self.fields['blurred_vision'].label = "Blurred Vision"
        
        # Secondary symptoms and complications
        self.fields['slow_healing'].label = "Slow Healing of Cuts/Wounds"
        self.fields['tingling'].label = "Numbness or Tingling in Hands/Feet"
        self.fields['skin_darkening'].label = "Dark Patches on Skin (Acanthosis Nigricans)"
        self.fields['frequent_infections'].label = "Frequent Infections (Urinary, Skin, etc.)"
        
        # Make all clinical measurements optional
        for field_name in [
            'height', 'weight', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'cholesterol', 'hdl_cholesterol', 'fasting_glucose', 'hba1c'
        ]:
            self.fields[field_name].required = False
