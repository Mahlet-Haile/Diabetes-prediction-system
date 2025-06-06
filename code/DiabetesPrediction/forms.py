from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import get_user_model
from .models import DiabetesAssessment, Patient, DiabetesUser

User = get_user_model()

# Authentication Forms
class DiabetesUserRegistrationForm(UserCreationForm):
    """Form for user registration"""
    email = forms.EmailField(max_length=255, required=True, widget=forms.EmailInput(
        attrs={'class': 'form-control', 'placeholder': 'Email'}))
    first_name = forms.CharField(max_length=100, required=False, widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'First Name'}))
    last_name = forms.CharField(max_length=100, required=False, widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Last Name'}))
    date_of_birth = forms.DateField(required=True, widget=forms.DateInput(
        attrs={'class': 'form-control', 'type': 'date', 'required': 'required'}))
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Password'}))
    password2 = forms.CharField(label='Confirm Password', widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Confirm Password'}))
    
    # Health profile fields
    has_diabetes_history = forms.BooleanField(required=False, widget=forms.CheckboxInput(
        attrs={'class': 'form-check-input'}),
        label="Do you have a personal history of diabetes?")
    has_family_diabetes_history = forms.BooleanField(required=False, widget=forms.CheckboxInput(
        attrs={'class': 'form-check-input'}),
        label="Do you have a family history of diabetes?")
    
    class Meta:
        model = User
        fields = ['email', 'first_name', 'last_name', 'date_of_birth', 
                 'password1', 'password2']
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already in use.")
        return email
        

        
    def save(self, commit=True):
        user = super().save(commit=False)
        # Set username from email before anything else
        user.username = self.cleaned_data.get('email')
        user.role = 'DOCTOR'  # Set default role to DOCTOR
        if commit:
            user.save()
        return user


class SuperAdminRegistrationForm(UserCreationForm):
    """Form for superadmin user registration (only for initial setup)"""
    email = forms.EmailField(max_length=255, required=True, widget=forms.EmailInput(
        attrs={'class': 'form-control', 'placeholder': 'Email'}))
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Password'}))
    password2 = forms.CharField(label='Confirm Password', widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Confirm Password'}))
    superadmin_code = forms.CharField(max_length=20, required=True, widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Superadmin Code'}),
        help_text="Enter the superadministrator code to verify your authorization.")    
    
    class Meta:
        model = User
        fields = ['email', 'password1', 'password2', 'superadmin_code']
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already in use.")
        return email
        
    def clean_superadmin_code(self):
        superadmin_code = self.cleaned_data.get('superadmin_code')
        if superadmin_code != "SUPERADMIN456":
            raise forms.ValidationError("Invalid superadmin code. Please enter the correct code.")
        return superadmin_code
        
    def save(self, commit=True):
        user = super().save(commit=False)
        user.is_staff = True
        user.is_superuser = True
        user.role = 'SUPERADMIN'
        if commit:
            user.save()
        return user


class AdminRegistrationForm(UserCreationForm):
    """Form for admin user registration"""
    email = forms.EmailField(max_length=255, required=True, widget=forms.EmailInput(
        attrs={'class': 'form-control', 'placeholder': 'Email'}))
    first_name = forms.CharField(max_length=100, required=False, widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'First Name'}))
    last_name = forms.CharField(max_length=100, required=False, widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Last Name'}))
    date_of_birth = forms.DateField(required=True, widget=forms.DateInput(
        attrs={'class': 'form-control', 'type': 'date', 'required': 'required'}))
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Password'}))
    password2 = forms.CharField(label='Confirm Password', widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Confirm Password'}))
    
    class Meta:
        model = User
        fields = ['email', 'first_name', 'last_name', 'date_of_birth', 'password1', 'password2']
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already in use.")
        return email
        

        
    def save(self, commit=True):
        user = super().save(commit=False)
        # Set username from email before anything else
        user.username = self.cleaned_data.get('email')
        user.is_staff = True
        user.role = 'ADMIN'
        if commit:
            user.save()
        return user


class DoctorRegistrationForm(UserCreationForm):
    """Form for doctor user registration"""
    email = forms.EmailField(max_length=255, required=True, widget=forms.EmailInput(
        attrs={'class': 'form-control', 'placeholder': 'Email'}))
    first_name = forms.CharField(max_length=100, required=True, widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'First Name'}))
    last_name = forms.CharField(max_length=100, required=True, widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Last Name'}))
    date_of_birth = forms.DateField(required=True, widget=forms.DateInput(
        attrs={'class': 'form-control', 'type': 'date', 'required': 'required'}))
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Password'}))
    password2 = forms.CharField(label='Confirm Password', widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Confirm Password'}))
    
    class Meta:
        model = User
        fields = ['email', 'first_name', 'last_name', 'date_of_birth', 'password1', 'password2', 'registered_by']
    
    def __init__(self, *args, **kwargs):
        self.registering_user = kwargs.pop('registering_user', None)
        super(DoctorRegistrationForm, self).__init__(*args, **kwargs)
        # Hide registered_by field as it's handled automatically
        if 'registered_by' in self.fields:
            del self.fields['registered_by']
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already in use.")
        return email
        

        
    def save(self, commit=True):
        user = super().save(commit=False)
        user.is_staff = False
        user.role = 'DOCTOR'
        if self.registering_user:
            user.registered_by = self.registering_user
        if commit:
            user.save()
        return user


class DiabetesUserLoginForm(AuthenticationForm):
    """Form for user login"""
    username = forms.EmailField(label='Email', max_length=255, widget=forms.EmailInput(
        attrs={'class': 'form-control', 'placeholder': 'Email'}))
    password = forms.CharField(label='Password', widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Password'}))
    
    class Meta:
        model = User
        fields = ['email', 'password']


# Patient Form


class PatientReassignmentForm(forms.Form):
    """Form for reassigning patients from one doctor to another"""
    patient = forms.ModelChoiceField(
        queryset=None,
        required=True,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label="Select Patient to Reassign"
    )
    new_doctor = forms.ModelChoiceField(
        queryset=None,
        required=True,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label="Assign to Doctor"
    )
    reason = forms.CharField(
        max_length=500,
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        label="Reason for Reassignment (Optional)"
    )
    
    def __init__(self, *args, **kwargs):
        # Get current doctor and user from kwargs
        current_doctor = kwargs.pop('current_doctor', None)
        current_user = kwargs.pop('current_user', None)
        super(PatientReassignmentForm, self).__init__(*args, **kwargs)
        
        # Set the queryset for patient selection based on the current doctor
        if current_doctor:
            self.fields['patient'].queryset = Patient.objects.filter(doctor=current_doctor)
        else:
            self.fields['patient'].queryset = Patient.objects.none()
        
        # Set the queryset for new doctor selection based on user role
        User = get_user_model()
        if current_user and current_user.role == 'SUPERADMIN':
            # Superadmin can see all doctors except current one
            doctors = User.objects.filter(role='DOCTOR')
        else:
            # Admin can only see doctors they registered
            doctors = User.objects.filter(role='DOCTOR', registered_by=current_user)
        
        # Exclude current doctor from options
        self.fields['new_doctor'].queryset = doctors.exclude(id=current_doctor.id if current_doctor else 0)


class PatientForm(forms.ModelForm):
    """Form for creating new patients"""
    full_name = forms.CharField(max_length=255, required=True, widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Patient Full Name'}))
    
    class Meta:
        model = Patient
        fields = ['full_name']
        # The patient_id and doctor fields are handled automatically


# Assessment Form
class DiabetesAssessmentForm(forms.ModelForm):
    """Form for diabetes assessment using only NHANES model features"""
    
    # Field for selecting an existing patient
    patient_selection = forms.ModelChoiceField(queryset=None, required=False, empty_label="Select Existing Patient", 
                                              widget=forms.Select(attrs={'class': 'form-control'}))
    
    # Field for entering a new patient's name (alternative to selecting existing patient)
    new_patient_name = forms.CharField(max_length=255, required=False, 
                                      widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter New Patient Name'}))
    
    class Meta:
        model = DiabetesAssessment
        # Only include fields that are used by the NHANES model
        fields = [
            # Patient information
            'patient_selection', 'new_patient_name',
            
            # Basic demographics 
            'gender', 'age',
            
            # Anthropometrics
            'height', 'weight',
            
            # Clinical measurements
            'fasting_glucose', 'hba1c', # glucose field removed as it's not in the NHANES dataset
            'blood_pressure_systolic', 'blood_pressure_diastolic',
            'cholesterol',
            
            # Risk factors
            'smoking_history', 'hypertension', 'heart_disease',
            
            # Symptoms
            'polyuria', 'polydipsia', 'polyphagia', 'weight_loss',
            'fatigue', 'blurred_vision', 'slow_healing', 'tingling'
        ]
        
        # Define widgets for better form rendering
        widgets = {
            # Basic demographics
            'gender': forms.Select(attrs={'class': 'form-control'}),
            'age': forms.NumberInput(attrs={'class': 'form-control', 'min': '1', 'max': '120'}),
            
            # Anthropometrics
            'height': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '50', 'max': '250', 'placeholder': 'cm'}),
            'weight': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '20', 'max': '500', 'placeholder': 'kg'}),
            
            # Clinical measurements
            'cholesterol': forms.Select(attrs={'class': 'form-control'}),
            # 'glucose' field removed as it's not in the NHANES dataset
            'fasting_glucose': forms.NumberInput(attrs={'class': 'form-control', 'step': '1', 'min': '50', 'max': '500', 'placeholder': 'mg/dL'}),
            'hba1c': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': '4', 'max': '14', 'placeholder': '%'}),
            'blood_pressure_systolic': forms.NumberInput(attrs={'class': 'form-control', 'min': '70', 'max': '250', 'placeholder': 'mmHg'}),
            'blood_pressure_diastolic': forms.NumberInput(attrs={'class': 'form-control', 'min': '40', 'max': '150', 'placeholder': 'mmHg'}),
            
            # Risk factors
            'smoking_history': forms.Select(attrs={'class': 'form-control'}),
            'hypertension': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'heart_disease': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            
            # Symptoms - based on NHANES dataset
            'polyuria': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'polydipsia': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'polyphagia': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'weight_loss': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'fatigue': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'blurred_vision': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'slow_healing': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'tingling': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def __init__(self, *args, **kwargs):
        # Get the doctor user from kwargs to filter patients
        self.doctor = kwargs.pop('doctor', None)  # Remove from kwargs to avoid passing to super
        super(DiabetesAssessmentForm, self).__init__(*args, **kwargs)
        
        # Set the queryset for patient selection based on the doctor
        if self.doctor:
            self.fields['patient_selection'].queryset = Patient.objects.filter(doctor=self.doctor)
        else:
            self.fields['patient_selection'].queryset = Patient.objects.none()
        
        # Add help text and labels for NHANES model fields
        # Demographics
        self.fields['gender'].label = "Gender"
        self.fields['age'].label = "Age"
        
        # Anthropometrics
        self.fields['height'].help_text = "Height in centimeters"
        self.fields['weight'].help_text = "Weight in kilograms"
        
        # Clinical measurements
        self.fields['cholesterol'].label = "Total Cholesterol Level"
        self.fields['glucose'].label = "Random Glucose Level"
        self.fields['fasting_glucose'].help_text = "Fasting Blood Glucose in mg/dL (if known)"
        self.fields['hba1c'].help_text = "Glycated Hemoglobin (HbA1c) in % (if known)"
        self.fields['blood_pressure_systolic'].help_text = "Systolic blood pressure (upper number)"
        self.fields['blood_pressure_diastolic'].help_text = "Diastolic blood pressure (lower number)"
        
        # Risk factors
        self.fields['smoking_history'].label = "Smoking History"
        self.fields['hypertension'].label = "Diagnosed with hypertension (high blood pressure)"
        self.fields['heart_disease'].label = "Diagnosed with heart disease (cardiovascular condition)"
        
        # Symptoms
        self.fields['polyuria'].label = "Frequent urination"
        self.fields['polydipsia'].label = "Excessive thirst"
        self.fields['polyphagia'].label = "Excessive hunger"
        self.fields['weight_loss'].label = "Unexplained weight loss"
        self.fields['fatigue'].label = "Unusual fatigue or tiredness"
        self.fields['blurred_vision'].label = "Blurred vision"
        self.fields['slow_healing'].label = "Cuts/sores that are slow to heal"
        self.fields['tingling'].label = "Numbness or tingling in hands/feet"
        
        # Make all clinical measurements optional
        self.fields['height'].required = False
        self.fields['weight'].required = False
        self.fields['cholesterol'].required = False
        self.fields['glucose'].required = False
        self.fields['fasting_glucose'].required = False
        self.fields['hba1c'].required = False
        self.fields['blood_pressure_systolic'].required = False
        self.fields['blood_pressure_diastolic'].required = False


# User Profile Update Form
class UserProfileUpdateForm(forms.ModelForm):
    """Form for updating user profile information"""
    email = forms.EmailField(max_length=255, required=True, widget=forms.EmailInput(
        attrs={'class': 'form-control', 'placeholder': 'Email'}))
    first_name = forms.CharField(max_length=100, required=False, widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'First Name'}))
    last_name = forms.CharField(max_length=100, required=False, widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Last Name'}))
    date_of_birth = forms.DateField(required=True, widget=forms.DateInput(
        attrs={'class': 'form-control', 'type': 'date', 'required': 'required'}))
    
    # Health profile fields
    has_diabetes_history = forms.BooleanField(required=False, widget=forms.CheckboxInput(
        attrs={'class': 'form-check-input'}),
        label="Do you have a personal history of diabetes?")
    has_family_diabetes_history = forms.BooleanField(required=False, widget=forms.CheckboxInput(
        attrs={'class': 'form-check-input'}),
        label="Do you have a family history of diabetes?")
    
    class Meta:
        model = User
        fields = ['email', 'first_name', 'last_name', 'date_of_birth']
    
    def __init__(self, *args, **kwargs):
        super(UserProfileUpdateForm, self).__init__(*args, **kwargs)
        # Make email field read-only if the user already exists
        if self.instance and self.instance.pk:
            self.initial['email'] = self.instance.email
            self.fields['email'].widget.attrs['readonly'] = True
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if self.instance and self.instance.pk:
            # For existing users, keep their current email
            return self.instance.email
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already in use.")
        return email
        


# Password Change Form
class PasswordChangeForm(forms.Form):
    """Form for changing user password"""
    current_password = forms.CharField(widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Current Password'}))
    new_password1 = forms.CharField(label='New Password', widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'New Password'}))
    new_password2 = forms.CharField(label='Confirm New Password', widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Confirm New Password'}))
    
    def __init__(self, user, *args, **kwargs):
        self.user = user
        super(PasswordChangeForm, self).__init__(*args, **kwargs)
    
    def clean_current_password(self):
        current_password = self.cleaned_data.get('current_password')
        if not self.user.check_password(current_password):
            raise forms.ValidationError("Your current password is incorrect.")
        return current_password
    
    def clean_new_password2(self):
        password1 = self.cleaned_data.get('new_password1')
        password2 = self.cleaned_data.get('new_password2')
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("The new passwords do not match.")
        return password2
