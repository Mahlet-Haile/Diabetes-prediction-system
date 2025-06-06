"""Views for NHANES data integration with the Diabetes Prediction System"""
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import logging

from .models import DiabetesAssessment
from .nhanes_forms import NHANESAssessmentForm

# Configure logging
logger = logging.getLogger(__name__)

@login_required
def nhanes_assessment_form(request):
    """View for NHANES-specific assessment form"""
    if not request.user.is_staff:
        messages.error(request, 'You do not have permission to access NHANES assessment.')
        return redirect('diabetes:dashboard')
        
    if request.method == 'POST':
        form = NHANESAssessmentForm(request.POST)
        if form.is_valid():
            assessment = form.save(commit=False)
            assessment.doctor = request.user
            assessment.save()
            messages.success(request, 'NHANES assessment completed successfully.')
            return redirect('diabetes:assessment_result', assessment_id=assessment.id)
    else:
        form = NHANESAssessmentForm()
    
    return render(request, 'DiabetesPrediction/nhanes_assessment_form.html', {
        'form': form,
        'title': 'NHANES Assessment Form'
    })

@login_required
def nhanes_assessment_form(request):
    """View for NHANES-specific assessment form"""
    # Restrict assessment access to doctors only
    if request.user.role != 'DOCTOR':
        messages.error(request, 'Only doctors are allowed to perform patient assessments.')
        return redirect('diabetes:dashboard')
    from .nhanes_forms import NHANESAssessmentForm
    from .models import Patient
    from .nhanes_ml_models_simplified import NHANESDiabetesPredictionModel, nhanes_model
    
    # Set NHANES model as active
    request.session['use_nhanes_model'] = True
    
    if request.method == 'POST':
        form = NHANESAssessmentForm(request.POST, doctor=request.user)
        
        if form.is_valid():
            assessment = form.save(commit=False)
            
            # Handle patient assignment
            if form.cleaned_data['patient_selection']:
                # Use selected patient
                assessment.patient = form.cleaned_data['patient_selection']
            elif form.cleaned_data['new_patient_name']:
                # Create new patient
                new_patient = Patient(
                    full_name=form.cleaned_data['new_patient_name'],
                    doctor=request.user
                )
                new_patient.save()
                assessment.patient = new_patient
            
            # Set doctor
            assessment.doctor = request.user
            assessment.save()
            
            # Process data and make prediction
            # Use the imported model instance instead of creating a new one
            if not nhanes_model._load_models():
                messages.error(request, 'NHANES prediction models not found. Please train them first.')
                return redirect('diabetes:dashboard')
            
            # Convert form data to format expected by model
            assessment_data = {
                'gender': assessment.gender,
                'age': assessment.age,
                'height': assessment.height,
                'weight': assessment.weight,
                'cholesterol': assessment.cholesterol,
                'glucose': assessment.glucose,
                'fasting_glucose': assessment.fasting_glucose,
                'hba1c': assessment.hba1c,
                'blood_pressure_systolic': assessment.blood_pressure_systolic,
                'blood_pressure_diastolic': assessment.blood_pressure_diastolic,
                'smoking_history': assessment.smoking_history,
                'hypertension': assessment.hypertension,
                'heart_disease': assessment.heart_disease,
                'polyuria': assessment.polyuria,
                'polydipsia': assessment.polydipsia,
                'polyphagia': assessment.polyphagia,
                'weight_loss': assessment.weight_loss,
                'fatigue': assessment.fatigue,
                'blurred_vision': assessment.blurred_vision,
                'slow_healing': assessment.slow_healing,
                'tingling': assessment.tingling,
            }
            
            # Make prediction using NHANES model
            prediction_result = nhanes_model.predict(assessment_data)
            
            # Update assessment with prediction results
            assessment.diabetes = prediction_result['has_diabetes']
            assessment.prediction_probability = prediction_result['diabetes_probability']
            assessment.diabetes_type = prediction_result['diabetes_type']
            assessment.confidence_score = prediction_result['confidence_score']
            assessment.save()
            
            # Redirect to results
            messages.success(request, 'Assessment completed successfully using NHANES model.')
            return redirect('diabetes:assessment_result', assessment_id=assessment.id)
        
        else:
            messages.error(request, 'There was an error with your submission. Please check the form and try again.')
    
    else:
        # Display empty form
        form = NHANESAssessmentForm(doctor=request.user)
    
    context = {
        'form': form,
        'page_title': 'NHANES Diabetes Assessment',
        'is_nhanes_form': True
    }
    
    return render(request, 'DiabetesPrediction/nhanes_assessment_form.html', context)
