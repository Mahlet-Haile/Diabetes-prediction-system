from django.db import models
from django.db.models import Max
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.utils import timezone
from django.conf import settings
from cryptography.fernet import Fernet

# User Authentication Models
class DiabetesUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        """Create and save a regular user"""
        if not email:
            raise ValueError('Users must have an email address')
            
        email = self.normalize_email(email)
        # Use email as username
        username = email.split('@')[0]
        user = self.model(email=email, username=username, **extra_fields)
        user.set_password(password)
        
        # If role is specified, set is_staff based on role
        role = extra_fields.get('role', 'DOCTOR')
        if role in ['SUPERADMIN', 'ADMIN']:
            user.is_staff = True
        
        user.save(using=self._db)
        return user
    
    def create_admin(self, email, username, password=None, **extra_fields):
        """Create and save an admin user"""
        extra_fields.setdefault('role', 'ADMIN')
        extra_fields.setdefault('is_staff', True)
        
        return self.create_user(email, username, password, **extra_fields)
    
    def create_doctor(self, email, username, password=None, **extra_fields):
        """Create and save a doctor user"""
        extra_fields.setdefault('role', 'DOCTOR')
        extra_fields.setdefault('is_staff', False)
        
        return self.create_user(email, username, password, **extra_fields)
        
    def create_superuser(self, email, username, password=None, **extra_fields):
        """Create and save a superuser"""
        extra_fields.setdefault('role', 'SUPERADMIN')
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True')
            
        return self.create_user(email, username, password, **extra_fields)


class DiabetesUser(AbstractBaseUser, PermissionsMixin):
    """Custom user model for Diabetes Prediction System"""
    email = models.EmailField(max_length=255, unique=True)
    username = models.CharField(max_length=100)
    first_name = models.CharField(max_length=100, blank=True)
    last_name = models.CharField(max_length=100, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    
    # User roles
    ROLE_CHOICES = [
        ('SUPERADMIN', 'Super Administrator'),
        ('ADMIN', 'Administrator'),
        ('DOCTOR', 'Doctor'),
    ]
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='DOCTOR')
    
    # Track which admin registered this user (primarily for doctors)
    registered_by = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, 
                                    related_name='registered_users')
    
    # User type and status fields
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    date_joined = models.DateTimeField(default=timezone.now)
    
    # Health profile fields
    has_diabetes_history = models.BooleanField(default=False)
    has_family_diabetes_history = models.BooleanField(default=False)
    
    objects = DiabetesUserManager()
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    
    def __str__(self):
        return self.email
        
    def get_full_name(self):
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username
        
    def get_short_name(self):
        return self.username


class Patient(models.Model):
    """Model for tracking patients with doctor-specific IDs"""
    # Doctor who added the patient
    doctor = models.ForeignKey(DiabetesUser, on_delete=models.CASCADE, related_name='patients')
    
    # Patient details
    full_name = models.CharField(max_length=255)
    patient_id = models.CharField(max_length=10)  # Format will be 0001, 0002, etc. per doctor
    date_added = models.DateTimeField(default=timezone.now)
    highest_diagnosis = models.CharField(max_length=20, default='low_risk')
    
    class Meta:
        unique_together = ('doctor', 'patient_id')  # Ensure IDs are unique per doctor
        ordering = ['doctor', 'patient_id']
        verbose_name = 'Patient'
        verbose_name_plural = 'Patients'
    
    def __str__(self):
        return f"{self.full_name} (ID: {self.patient_id}, Doctor: {self.doctor.username})"
    
    def save(self, *args, **kwargs):
        # Generate patient_id if not provided
        if not self.patient_id:
            # Get the highest patient_id for this doctor
            last_patient = Patient.objects.filter(doctor=self.doctor).order_by('-patient_id').first()
            if last_patient:
                try:
                    last_num = int(last_patient.patient_id)
                    self.patient_id = f"{last_num + 1:04d}"
                except ValueError:
                    # If patient_id is not numeric, start from 0001
                    self.patient_id = "0001"
            else:
                self.patient_id = "0001"
        
        # Generate unique patient_id for this doctor
        while Patient.objects.filter(doctor=self.doctor, patient_id=self.patient_id).exists():
            try:
                current_num = int(self.patient_id)
                self.patient_id = f"{current_num + 1:04d}"
            except ValueError:
                # If patient_id is not numeric, start from 0001
                self.patient_id = "0001"

        # Set default values for required fields
        if not hasattr(self, 'highest_diagnosis'):
            self.highest_diagnosis = 'low_risk'
        
        super().save(*args, **kwargs)


class EncryptedField(models.TextField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fernet = Fernet(settings.ENCRYPTION_KEY)

    def from_db_value(self, value, expression=None, connection=None):
        if value is None:
            return value
        try:
            # Decrypt the value
            decrypted = self.fernet.decrypt(value.encode())
            # Try to parse as JSON
            try:
                import json
                return json.loads(decrypted.decode())
            except json.JSONDecodeError:
                # If not JSON, return as string
                return decrypted.decode()
        except Exception as e:
            # If decryption fails, return the raw value
            return value

    def to_python(self, value):
        if value is None:
            return value
        if isinstance(value, (dict, list)):
            return value
        try:
            import json
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def get_prep_value(self, value):
        if value is None:
            return value
        # Convert to JSON if it's a dict or list
        if isinstance(value, (dict, list)):
            import json
            value = json.dumps(value)
        # Convert to string if it's not already
        elif not isinstance(value, str):
            value = str(value)
        # Encrypt the value
        encrypted = self.fernet.encrypt(value.encode())
        return encrypted.decode()

class DiabetesAssessment(models.Model):
    """
    Model for storing diabetes assessment data including both measured clinical values
    and non-measured symptoms
    """
    # User identification - keeping for backward compatibility during migration
    user = models.ForeignKey(DiabetesUser, on_delete=models.CASCADE, related_name='assessments', null=True, blank=True)
    
    # Patient and doctor identification 
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='assessments', null=True, blank=True)
    doctor = models.ForeignKey(DiabetesUser, on_delete=models.CASCADE, related_name='doctor_assessments', null=True, blank=True)
    assessment_date = models.DateTimeField(auto_now_add=True)
    
    # Demographic information
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female')
    ]
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES, null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    height = models.FloatField(help_text="Height in cm", null=True, blank=True)
    weight = models.FloatField(help_text="Weight in kg", null=True, blank=True)
    
    # Clinical measurements
    # Changed from categorical to numerical to align with NHANES dataset
    cholesterol = models.FloatField(null=True, blank=True, help_text="Total cholesterol in mg/dL")
    # glucose field removed as it's not in the NHANES dataset and could cause confusion
    fasting_glucose = models.FloatField(null=True, blank=True, help_text="Fasting blood glucose in mg/dL")
    hba1c = models.FloatField(null=True, blank=True, help_text="HbA1c percentage")
    blood_pressure_systolic = models.IntegerField(null=True, blank=True)
    blood_pressure_diastolic = models.IntegerField(null=True, blank=True)
    
    # Risk factors
    SMOKING_HISTORY_CHOICES = [
        ('not_smoker', 'Not a Smoker'),
        ('smoker', 'Smoker'),
        ('no_info', 'No Information')
    ]
    smoking_history = models.CharField(max_length=20, choices=SMOKING_HISTORY_CHOICES, default='no_info')
    smoking = models.BooleanField(default=False, help_text="Legacy field - use smoking_history instead")
    alcohol = models.BooleanField(default=False)
    active = models.BooleanField(default=False)
    physical_activity = models.IntegerField(null=True, blank=True, help_text="Minutes per week")
    family_history = models.BooleanField(default=False, help_text="Family history of diabetes")
    hypertension = models.BooleanField(default=False)
    heart_disease = models.BooleanField(default=False, help_text="History of cardiovascular disease")
    
    # Additional clinical measurements for comprehensive assessment
    hdl_cholesterol = models.IntegerField(null=True, blank=True, help_text="HDL (High-density lipoprotein) cholesterol in mg/dL")
    ldl_cholesterol = models.IntegerField(null=True, blank=True, help_text="LDL (Low-density lipoprotein) cholesterol in mg/dL")
    triglycerides = models.IntegerField(null=True, blank=True, help_text="Triglycerides level in mg/dL")
    
    # Symptoms (Type 2 Diabetes)
    polyuria = models.BooleanField(default=False, help_text="Frequent urination")
    polydipsia = models.BooleanField(default=False, help_text="Excessive thirst")
    polyphagia = models.BooleanField(default=False, help_text="Excessive hunger")
    weight_loss = models.BooleanField(default=False, help_text="Unexplained weight loss")
    fatigue = models.BooleanField(default=False)
    blurred_vision = models.BooleanField(default=False)
    slow_healing = models.BooleanField(default=False, help_text="Slow healing of cuts and wounds")
    tingling = models.BooleanField(default=False, help_text="Tingling or numbness in hands/feet")
    skin_darkening = models.BooleanField(default=False, help_text="Dark patches on skin (Acanthosis Nigricans)")
    frequent_infections = models.BooleanField(default=False, help_text="Frequent infections (urinary, skin, etc.)")
    
    # Complication warning signs
    chest_pain = models.BooleanField(default=False)
    shortness_of_breath = models.BooleanField(default=False)
    swelling_in_legs = models.BooleanField(default=False)
    numbness = models.BooleanField(default=False)
    foot_ulcers = models.BooleanField(default=False)
    vision_loss = models.BooleanField(default=False)
    
    # ML prediction results
    diabetes = models.BooleanField(default=False, null=True, blank=True)
    prediction_probability = models.FloatField(null=True, blank=True)
    
    DIABETES_TYPE_CHOICES = [
        ('none', 'No Diabetes'),
        ('prediabetes', 'Prediabetes'),
        ('type1', 'Type 1 Diabetes'),
        ('type2', 'Type 2 Diabetes'),
        ('gestational', 'Gestational Diabetes'),
    ]
    diabetes_type = models.CharField(max_length=15, choices=DIABETES_TYPE_CHOICES, default='none')
    confidence_score = models.FloatField(null=True, blank=True)  # Confidence in prediction
    
    # Expert system results - Encrypted for sensitive data
    risk_score = models.FloatField(null=True, blank=True)
    diagnosis = EncryptedField(null=True, blank=True)
    expert_explanation = EncryptedField(null=True, blank=True)
    recommendations = EncryptedField(null=True, blank=True)
    complication_risks = EncryptedField(null=True, blank=True)
    
    # Recommendations - Encrypted for patient privacy
    diet_recommendations = EncryptedField(null=True, blank=True)
    exercise_recommendations = EncryptedField(null=True, blank=True)
    monitoring_recommendations = EncryptedField(null=True, blank=True)
    
    @property
    def bmi(self):
        """Calculate BMI based on height and weight"""
        if self.height and self.weight:
            height_m = self.height / 100  # Convert cm to meters
            return round(self.weight / (height_m * height_m), 1)
        return None
        
    @property
    def bmi_category(self):
        """Return BMI category based on calculated BMI"""
        bmi = self.bmi
        if bmi is None:
            return None
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def __str__(self):
        return f"Assessment for Patient {self.patient.patient_id} ({self.patient.full_name}) on {self.assessment_date.strftime('%Y-%m-%d')}"
        
        
    class Meta:
        ordering = ['-assessment_date']
        verbose_name = 'Diabetes Assessment'
        verbose_name_plural = 'Diabetes Assessments'
