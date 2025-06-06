from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout, get_user_model
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect
from django.urls import reverse
from django.db import models
from django.db.models import Count, Q
from django.utils import timezone
from django.core.exceptions import PermissionDenied
from django.contrib.auth.hashers import make_password
from django.conf import settings
from cryptography.fernet import Fernet
from .forms import DiabetesUserRegistrationForm, DiabetesUserLoginForm, AdminRegistrationForm, DoctorRegistrationForm, PatientForm, SuperAdminRegistrationForm, PatientReassignmentForm, UserProfileUpdateForm, PasswordChangeForm
from .comprehensive_forms import ComprehensiveDiabetesAssessmentForm
from .models import DiabetesAssessment, DiabetesUser, Patient
# Import the NHANES model
from .nhanes_ml_models_simplified import NHANESDiabetesPredictionModel, nhanes_model
from .nhanes_data_integration import DiabetesPredictionIntegrator
from .expert_system.rule_engine import DiabetesExpertSystem, prepare_patient_data

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the prediction models
nhanes_model = NHANESDiabetesPredictionModel()

# Function to get the NHANES model for prediction
def get_active_model(request):
    """Get the NHANES model for prediction"""
    # Always use NHANES models
    integrator = DiabetesPredictionIntegrator(use_nhanes=True)
    
    # Return the NHANES model
    return integrator.get_active_model()

# Authentication Views
def register_view(request):
    """User registration view"""
    if request.user.is_authenticated:
        return redirect('diabetes:dashboard')
        
    if request.method == 'POST':
        form = DiabetesUserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful! Welcome to the Diabetes Prediction System.')
            return redirect('diabetes:dashboard')
    else:
        form = DiabetesUserRegistrationForm()
        
    return render(request, 'DiabetesPrediction/register.html', {'form': form})


from django.views.decorators.csrf import csrf_protect, ensure_csrf_cookie, csrf_exempt
from django.core.exceptions import PermissionDenied

@ensure_csrf_cookie  # Ensures the response sets a CSRF cookie
@csrf_protect      # Provides CSRF protection for this view
def login_view(request):
    """User login view with enhanced CSRF handling"""
    # Check if the re-login parameter is present
    if request.GET.get('relogin') and request.user.is_authenticated:
        # Log the current user out
        logout(request)
        # Add a message informing the user they've been logged out
        messages.info(request, 'You have been logged out. Please log in again.')
        # Redirect back to login without the parameter to avoid loops
        return redirect('diabetes:login')
    
    # Standard redirect if already authenticated
    elif request.user.is_authenticated:
        return redirect('diabetes:index')
    
    # CSRF token handling - regenerate CSRF token if it might be stale
    csrf_token = request.META.get('CSRF_COOKIE', None)
    
    # Initialize form variables
    form = None
    
    if request.method == 'POST':
        try:
            form = DiabetesUserLoginForm(request=request, data=request.POST)
            if form.is_valid():
                email = form.cleaned_data.get('username')  # Form uses 'username' field for email
                password = form.cleaned_data.get('password')
                user = authenticate(username=email, password=password)
                if user is not None:
                    login(request, user)
                    messages.success(request, f'Welcome back, {user.get_short_name()}!')
                    
                    # Redirect to requested page if available
                    next_page = request.GET.get('next')
                    if next_page:
                        return redirect(next_page)
                    return redirect('diabetes:index')
                else:
                    messages.error(request, 'Invalid email or password.')
            else:
                # Handle CSRF errors specially
                if 'CSRF' in str(form.errors):
                    # Create a new form with fresh token
                    form = DiabetesUserLoginForm()
                    messages.error(request, 'Your session has expired. Please try again with the refreshed form.')
                else:
                    messages.error(request, 'Invalid email or password.')
        except PermissionDenied:
            # This will catch CSRF validation failures
            messages.error(request, 'Security validation failed. The form has been refreshed, please try again.')
            form = DiabetesUserLoginForm()
    else:
        # Always give a fresh form for GET requests
        form = DiabetesUserLoginForm()
    
    # Prepare the response with cache control headers
    response = render(request, 'DiabetesPrediction/login.html', {
        'form': form,
        'csrf_token_rotated': csrf_token is None
    })
    
    # Set strict cache control headers to prevent any browser caching
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'  # HTTP 1.0
    response['Expires'] = '0'  # Proxies
    response['X-Content-Type-Options'] = 'nosniff'  # Prevent MIME type sniffing
    
    return response


def logout_view(request):
    """User logout view"""
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('diabetes:index')


# Home page view
def index(request):
    """Home page view"""
    return render(request, 'DiabetesPrediction/index.html')

# Dashboard view
def dashboard(request):
    # Check if user is authenticated
    if not request.user.is_authenticated:
        return redirect('diabetes:login')
    
    # If user is superadmin or admin, redirect to admin dashboard
    if request.user.is_staff:
        return redirect('diabetes:admin_dashboard')
    
    # For regular users and doctors, show the normal dashboard
    return render(request, 'DiabetesPrediction/dashboard.html', {
        'username': request.user.get_full_name() or request.user.username,
        'user': request.user,
    })

# Assessment form view
@login_required
def assessment_form(request):
    """Display the comprehensive assessment form"""
    # Restrict assessment access to doctors only
    if request.user.role != 'DOCTOR':
        messages.error(request, 'Only doctors are allowed to perform patient assessments.')
        return redirect('diabetes:dashboard')
    if request.method == 'POST':
        form = ComprehensiveDiabetesAssessmentForm(request.POST, doctor=request.user)
        if form.is_valid():
            # Create a new assessment
            assessment = form.save(commit=False)
            
            # Associate with doctor (current user)
            assessment.doctor = request.user
            
            # Process patient information
            patient_selection = form.cleaned_data.get('patient_selection')
            new_patient_name = form.cleaned_data.get('new_patient_name')
            new_patient_last_name = form.cleaned_data.get('new_patient_last_name')
            
            if patient_selection:
                # Use selected existing patient
                assessment.patient = patient_selection
            elif new_patient_name and new_patient_last_name:
                # Combine first and last name to create full name
                full_name = f"{new_patient_name} {new_patient_last_name}"
                
                # Create a new patient
                patient_data = {
                    'full_name': full_name,
                    'doctor': request.user
                }
                
                # Simple approach - if a new patient name is provided, always create a new patient
                # This honors the user's intent to create a new patient even if similar names exist
                # The warning on the frontend already alerted the user about potential duplicates
                
                patient = Patient(
                    doctor=request.user,
                    full_name=full_name
                )
                
                # Create a new patient using Django's ORM
                patient = Patient(
                    doctor=request.user,
                    full_name=full_name,
                    highest_diagnosis='low_risk'
                )
                patient.save()
                
                # The patient_id will be automatically generated by the save method
                logger.info(f"Created new patient with name: {full_name}, ID: {patient.patient_id}")
                
                created = True
                logger.info(f"Created new patient with name: {full_name}, ID: {patient.patient_id}")
                
                # Associate assessment with the new patient
                assessment.patient = patient
            else:
                # This shouldn't happen due to form validation, but handle it anyway
                messages.error(request, 'Please select an existing patient or enter both first and last name for a new patient.')
                return render(request, 'DiabetesPrediction/comprehensive_assessment_form.html', {'form': form})
            
            # BMI is automatically calculated as a property in the model
            # No need to calculate or set it manually
            
            # Save assessment
            assessment.save()
            
            # Get the prediction using the NHANES model
            integrator = DiabetesPredictionIntegrator(use_nhanes=True)
            
            # Convert assessment to NHANES format
            nhanes_data = integrator.convert_assessment_to_nhanes(assessment)
            
            # Get prediction
            prediction = integrator.predict(nhanes_data)
            
            # Prepare expert system data
            patient_data = prepare_patient_data(assessment)
            
            # Run expert system analysis
            expert_system = DiabetesExpertSystem(patient_data)
            expert_results = expert_system.run_assessment()
            
            # Merge ML prediction with expert system results
            if prediction:
                # Copy expert system diagnosis
                prediction['diagnosis'] = expert_results['diagnosis']
                prediction['risk_score'] = expert_results['risk_score']
                prediction['recommendations'] = expert_results['recommendations']
                
                # Save the explanation as well
                assessment.expert_explanation = '\n'.join(expert_results['explanation'])
            else:
                # Use only expert system results if ML model fails
                prediction = expert_results
                
                # Save the explanation directly
                assessment.expert_explanation = '\n'.join(expert_results['explanation'])
            
            # Update assessment with prediction
            updated_assessment = integrator.update_assessment_with_prediction(assessment, prediction)
            updated_assessment.save()
            
            # Redirect to result page
            return redirect('diabetes:assessment_result', assessment_id=assessment.id)
    else:
        # Display empty form
        form = ComprehensiveDiabetesAssessmentForm(doctor=request.user)
    
    messages.info(request, 'Using comprehensive assessment form with integrated NHANES model and expert system.')
    return render(request, 'DiabetesPrediction/comprehensive_assessment_form.html', {'form': form})

# Assessment result view
@login_required
def assessment_result(request, assessment_id):
    """View for showing the expert system assessment results"""
    import json
    try:
        # Ensure doctor can only see their patients' assessments
        assessment = DiabetesAssessment.objects.get(id=assessment_id, doctor=request.user)
        
        # Prepare context for result template
        context = {
            'username': request.user.username,
            'assessment': assessment,
            'patient': assessment.patient,
            'assessment_date': assessment.assessment_date.strftime('%Y-%m-%d %H:%M'),
            'bmi': assessment.bmi,
            'bmi_category': assessment.bmi_category,
        }
        
        # Add expert system results if available
        if assessment.diagnosis:
            try:
                context['diagnosis'] = assessment.diagnosis
                context['risk_score'] = float(assessment.risk_score) if assessment.risk_score else 0
                
                # Handle recommendations - they are already deserialized by EncryptedField
                if assessment.recommendations:
                    context['recommendations'] = assessment.recommendations
                else:
                    context['recommendations'] = {'diet': [], 'activity': [], 'lifestyle': [], 'medical': []}
                
                if assessment.diet_recommendations:
                    context['diet_recommendations'] = assessment.diet_recommendations
                else:
                    context['diet_recommendations'] = []
                
                if assessment.exercise_recommendations:
                    context['exercise_recommendations'] = assessment.exercise_recommendations
                else:
                    context['exercise_recommendations'] = []
                
                if assessment.monitoring_recommendations:
                    context['monitoring_recommendations'] = assessment.monitoring_recommendations
                else:
                    context['monitoring_recommendations'] = []
                
                if assessment.expert_explanation:
                    context['expert_explanation'] = assessment.expert_explanation
            except Exception as e:
                messages.error(request, f'Error processing assessment results: {str(e)}')
        
        # Add expert system results if available
        if assessment.diagnosis:
            context['diagnosis'] = assessment.diagnosis
            
            # Normalize risk scores to ensure diabetes always has higher risk than prediabetes
            # This is a display fix that matches the assessment_history view
            raw_risk_score = float(assessment.risk_score) if assessment.risk_score is not None else 0
            if assessment.diagnosis == 'diabetes':
                # Ensure diabetes always shows the highest risk (minimum 25)
                if raw_risk_score < 25:
                    display_risk_score = 25.0
                else:
                    display_risk_score = raw_risk_score
            elif assessment.diagnosis == 'prediabetes':
                # Ensure prediabetes shows appropriate risk (maximum 20)
                if raw_risk_score > 20:
                    display_risk_score = 20.0
                else:
                    display_risk_score = raw_risk_score
            else:
                display_risk_score = raw_risk_score
                
            context['risk_score'] = display_risk_score
            context['original_risk_score'] = raw_risk_score  # Store original for reference if needed
            context['expert_explanation'] = assessment.expert_explanation.split('\n') if assessment.expert_explanation else []
            
            # Load JSON recommendations if available
            if assessment.recommendations:
                if isinstance(assessment.recommendations, str):
                    context['recommendations'] = json.loads(assessment.recommendations)
                else:
                    context['recommendations'] = assessment.recommendations
            
            # Load JSON complication risks if available
            if assessment.complication_risks:
                if isinstance(assessment.complication_risks, str):
                    context['complication_risks'] = json.loads(assessment.complication_risks)
                else:
                    context['complication_risks'] = assessment.complication_risks
        
        # For backward compatibility, still provide risk_level
        if hasattr(assessment, 'diabetes_risk_score') and assessment.diabetes_risk_score is not None:
            context['risk_level'] = get_risk_level(assessment.diabetes_risk_score)
        elif assessment.risk_score is not None:
            context['risk_level'] = get_risk_level(assessment.risk_score)
        else:
            context['risk_level'] = 'Unknown'
        
        return render(request, "DiabetesPrediction/assessment_result.html", context)
        
    except DiabetesAssessment.DoesNotExist:
        messages.error(request, 'Assessment not found or unauthorized access.')
        return redirect('diabetes:dashboard')
    except Exception as e:
        logger.error(f"Error in assessment_result: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
@login_required
def assessment_history(request):
    """Display history of user's diabetes assessments grouped by patient"""
    # Only allow doctors to access history
    if request.user.role != 'DOCTOR':
        messages.error(request, 'Only doctors can access assessment history.')
        return redirect('diabetes:dashboard')
        
    try:
        # Get the patient_id filter from request parameters, if provided
        patient_id = request.GET.get('patient_id')
        search_query = request.GET.get('search_patient_id', '')
        
        if patient_id:
            # If a specific patient ID is provided, get their details and assessments
            try:
                # Get the specific patient with matching patient_id belonging to this doctor
                patient = Patient.objects.get(patient_id=patient_id, doctor=request.user)
                # Get all assessments for this specific patient, ordered by most recent first
                assessments = DiabetesAssessment.objects.filter(patient=patient).order_by('-assessment_date')
                # Set the view mode to show details for a specific patient
                view_mode = 'patient_detail'
            except Patient.DoesNotExist:
                # If the patient doesn't exist or doesn't belong to this doctor, redirect to the list view
                messages.error(request, f"Patient with ID {patient_id} not found.")
                return redirect('diabetes:assessment_history')
        else:
            # Show patients list, possibly filtered by search query
            if search_query:
                # Search for patients by patient ID (partial match)
                patients = Patient.objects.filter(
                    doctor=request.user,
                    patient_id__icontains=search_query
                ).order_by('patient_id')
                
                # If exactly one patient is found, redirect to that patient's details
                if patients.count() == 1:
                    return redirect(f"{request.path}?patient_id={patients.first().patient_id}")
                    
                # If no patients found, show a message
                if not patients.exists():
                    messages.info(request, f"No patients found with ID containing '{search_query}'")
            else:
                # If no search query, show all patients belonging to this doctor
                patients = Patient.objects.filter(doctor=request.user).order_by('full_name')
                
            assessments = []  # No assessments to show in the list view
            view_mode = 'patient_list'
            
        # Only process assessments if we're in patient detail view
        if patient_id and assessments:
            # Create a list to hold assessment data with computed values
            processed_assessments = []
            
            # Add risk level display to each assessment
            for assessment in assessments:
                # Create a dictionary to hold computed values
                data = {}
                
                # Add all assessment attributes to the data dictionary
                for field in assessment._meta.fields:
                    data[field.name] = getattr(assessment, field.name, None)
                
                # Add assessment instance for reference (to use in templates for links, etc.)
                data['assessment_obj'] = assessment
                
                # Calculate BMI if it's not already available
                if not hasattr(assessment, 'bmi') or not assessment.bmi:
                    if assessment.height and assessment.weight and assessment.height > 0:
                        # BMI formula: weight(kg) / (height(m) ^ 2)
                        height_in_meters = assessment.height / 100  # Convert cm to meters
                        data['bmi'] = assessment.weight / (height_in_meters * height_in_meters)
                    else:
                        data['bmi'] = 0
                else:
                    data['bmi'] = assessment.bmi
                    
                # Normalize risk scores for existing assessments to ensure diabetes always has higher risk than prediabetes
                # This is a display fix that doesn't alter the stored values
                risk_score = 0
                if hasattr(assessment, 'risk_score') and assessment.risk_score is not None:
                    try:
                        risk_score = float(assessment.risk_score)
                    except (ValueError, TypeError):
                        risk_score = 0
                        
                if assessment.diagnosis == 'diabetes':
                    # Ensure diabetes always shows the highest risk (minimum 25)
                    if risk_score < 25:
                        data['display_risk_score'] = 25.0
                    else:
                        data['display_risk_score'] = risk_score
                    data['risk_level'] = 'Diabetes'
                    data['risk_badge_color'] = 'bg-danger'
                    data['diagnosis'] = 'diabetes'  # Ensure consistency
                elif assessment.diagnosis == 'prediabetes':
                    # Ensure prediabetes shows appropriate risk (maximum 20)
                    if risk_score > 20:
                        data['display_risk_score'] = 20.0
                    else:
                        data['display_risk_score'] = risk_score
                    data['risk_level'] = 'Prediabetes'
                    data['risk_badge_color'] = 'bg-warning text-dark'
                    data['diagnosis'] = 'prediabetes'  # Ensure consistency
                # Handle non-diabetic and pre-diabetic risk categories - first set the risk score
                elif assessment.diagnosis in ['low_risk', 'moderate_risk', 'high_risk', 'very_high_risk'] or getattr(assessment, 'diabetes_type', 'none') == 'none':
                    # For non-diabetic patients, show appropriate risk level
                    data['display_risk_score'] = risk_score
                    # risk_score is already set above
                    
                    # Determine risk level based on score for non-diabetic patients
                    diagnosis = assessment.diagnosis
                    if risk_score < 5:
                        data['risk_level'] = 'Low'
                        data['risk_badge_color'] = 'bg-success'
                        # If diagnosis is empty but we calculated a risk level
                        if not diagnosis or diagnosis == 'unknown':
                            data['diagnosis'] = 'low_risk'
                    elif risk_score < 10:
                        data['risk_level'] = 'Moderate'
                        data['risk_badge_color'] = 'bg-warning text-dark'
                        if not diagnosis or diagnosis == 'unknown':
                            data['diagnosis'] = 'moderate_risk'
                    elif risk_score < 15:
                        data['risk_level'] = 'High'
                        data['risk_badge_color'] = 'bg-orange'
                        if not diagnosis or diagnosis == 'unknown':
                            data['diagnosis'] = 'high_risk'
                    else:
                        data['risk_level'] = 'Very High'
                        data['risk_badge_color'] = 'bg-danger'
                        if not diagnosis or diagnosis == 'unknown':
                            data['diagnosis'] = 'very_high_risk'
                            
                    # Override specific diagnoses with their exact names
                    if diagnosis == 'low_risk':
                        data['risk_level'] = 'Low'
                        data['risk_badge_color'] = 'bg-success'
                    elif diagnosis == 'moderate_risk':
                        data['risk_level'] = 'Moderate'
                        data['risk_badge_color'] = 'bg-warning text-dark'
                    elif diagnosis == 'high_risk':
                        data['risk_level'] = 'High'
                        data['risk_badge_color'] = 'bg-orange'
                    elif diagnosis == 'very_high_risk':
                        data['risk_level'] = 'Very High'
                        data['risk_badge_color'] = 'bg-danger'
                else:
                    # Fallback for any other scenario
                    data['display_risk_score'] = risk_score
                    if risk_score < 5:
                        data['risk_level'] = 'Low'
                        data['risk_badge_color'] = 'bg-success'
                    elif risk_score < 10:
                        data['risk_level'] = 'Moderate'
                        data['risk_badge_color'] = 'bg-warning text-dark'
                    elif risk_score < 15:
                        data['risk_level'] = 'High'
                        data['risk_badge_color'] = 'bg-orange'
                    else:
                        data['risk_level'] = 'Very High'
                        data['risk_badge_color'] = 'bg-danger'
                        
                # Simplified diabetes type handling - no longer distinguishing between Type 1 and Type 2
                if assessment.diagnosis == 'diabetes':
                    data['diabetes_type'] = 'diabetes'
                elif assessment.diagnosis == 'prediabetes':
                    data['diabetes_type'] = 'prediabetes'
                else:
                    data['diabetes_type'] = 'none'
                    
                # Add the processed assessment data to our list
                processed_assessments.append(data)
        
        # Prepare the context data based on view mode
        context = {
            'view_mode': view_mode,
            'search_query': search_query  # Add search query to context for displaying in the form
        }
        
        if view_mode == 'patient_list':
            context['patients'] = patients
        else:  # patient_detail mode
            context['patient'] = patient
            # Use the processed assessments with computed values instead of database objects
            context['assessments'] = processed_assessments if 'processed_assessments' in locals() else []
        
        return render(request, 'DiabetesPrediction/assessment_history.html', context)
        
    except Exception as e:
        # Log more detailed error information
        import traceback
        logger.error(f"Error in assessment_history: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Get view mode and parameters for debugging
        patient_id = request.GET.get('patient_id', 'None')
        logger.error(f"Parameters: patient_id={patient_id}")
        
        # Show more specific error message to help debugging
        messages.error(request, f"An error occurred while retrieving assessment history: {str(e)}")
        return redirect('diabetes:dashboard')

# Helper function to determine risk level from risk score
def get_risk_level(risk_score):
    """Convert numerical risk score to categorical risk level"""
    if risk_score is None:
        return "Unknown"
        
    try:
        score = float(risk_score)
    except (ValueError, TypeError):
        return "Unknown"
        
    # Updated risk categorization with clearer thresholds    
    if score < 10:
        return "Low"
    elif score < 20:
        return "Moderate"
    elif score < 26:
        return "High"
    else:
        return "Very High"

# Get patient data from previous assessment for auto-fill
@login_required
def get_patient_data(request):
    """Get patient data from previous assessments for auto-fill"""
    try:
        patient_id = request.GET.get('patient_id')
        if not patient_id:
            return JsonResponse({'error': 'No patient ID provided'}, status=400)
            
        # Add debug logging
        logger.info(f"Attempting to fetch patient data for patient_id: {patient_id}")
        
        # Always return success=True by default
        data = {
            'success': True,
            'gender': '',
            'age': ''
        }
        
        # Try different ways to find the patient
        patient = None
        
        # First try with numeric ID
        try:
            patient = Patient.objects.get(id=patient_id, doctor=request.user)
            logger.info(f"Found patient with ID {patient_id}: {patient.full_name}")
        except (Patient.DoesNotExist, ValueError):
            # Then try with patient_id field
            try:
                patient = Patient.objects.get(patient_id=patient_id, doctor=request.user)
                logger.info(f"Found patient with patient_id {patient_id}: {patient.full_name}")
            except Patient.DoesNotExist:
                try:
                    # Last attempt - maybe patient_id is the full database ID
                    assessments = DiabetesAssessment.objects.filter(id=patient_id, doctor=request.user)
                    if assessments.exists():
                        patient = assessments.first().patient
                        logger.info(f"Found patient through assessment ID {patient_id}")
                except Exception:
                    logger.error(f"Patient not found for ID {patient_id}")
                    return JsonResponse({'error': 'Patient not found', 'success': False}, status=404)
        
        if not patient:
            return JsonResponse({'error': 'Patient not found', 'success': False}, status=404)
            
        # Get the most recent assessment for this patient
        assessments = DiabetesAssessment.objects.filter(patient=patient).order_by('-assessment_date')
        logger.info(f"Found {assessments.count()} assessments for patient {patient.full_name}")
        
        if assessments.exists():
            latest_assessment = assessments.first()
            
            # Update data with patient information
            if latest_assessment.gender:
                data['gender'] = latest_assessment.gender
                logger.info(f"Using gender from assessment: {data['gender']}")
                
            if latest_assessment.age:
                data['age'] = latest_assessment.age
                logger.info(f"Using age from assessment: {data['age']}")
        else:
            logger.info(f"No assessments found for patient {patient.full_name}")
            # Even without assessments, we return success=True with empty values
        
        logger.info(f"Returning patient data: {data}")
        return JsonResponse(data)
        
    except Exception as e:
        logger.error(f"Error in get_patient_data: {str(e)}")
        return JsonResponse({'error': str(e), 'success': False}, status=500)

# Delete patient view
@login_required
def delete_patient(request, patient_id):
    """Delete a patient and all associated assessments"""
    try:
        # Ensure doctor can only delete their own patients
        patient = get_object_or_404(Patient, id=patient_id, doctor=request.user)
        
        # Get patient name for the success message
        patient_name = patient.full_name
        patient_id_value = patient.patient_id
        
        # Delete the patient (this will cascade delete all related assessments)
        patient.delete()
        
        messages.success(request, f'Patient {patient_name} (ID: {patient_id_value}) and all associated assessments have been deleted.')
    except Exception as e:
        logger.error(f"Error in delete_patient: {e}")
        messages.error(request, f"An error occurred while deleting the patient: {str(e)}")
    
    # Redirect to assessment history (patient list)
    return redirect('diabetes:assessment_history')

# Delete assessment view
@login_required
def delete_assessment(request, assessment_id):
    """Delete an assessment and also delete the patient if it was their last assessment"""
    try:
        # Ensure user can only delete their assessments
        assessment = DiabetesAssessment.objects.get(id=assessment_id, doctor=request.user)
        
        # Store patient reference before deleting assessment
        patient = assessment.patient
        patient_name = patient.full_name
        patient_id_value = patient.patient_id
        
        # Delete the assessment
        assessment.delete()
        messages.success(request, 'Assessment deleted successfully.')
        
        # Check if this was the last assessment for this patient
        remaining_assessments = DiabetesAssessment.objects.filter(patient=patient).count()
        if remaining_assessments == 0:
            # This was the last assessment, also delete the patient
            patient.delete()
            messages.info(request, f'Patient {patient_name} (ID: {patient_id_value}) has been automatically removed as they have no remaining assessments.')
            
    except DiabetesAssessment.DoesNotExist:
        messages.error(request, 'Assessment not found or unauthorized access.')
    except Exception as e:
        logger.error(f"Error in delete_assessment: {e}")
        messages.error(request, f"An error occurred: {str(e)}")
    
    # Redirect to previous page if available, or fall back to assessment history
    if request.META.get('HTTP_REFERER'):
        return redirect(request.META.get('HTTP_REFERER'))
    else:
        return redirect('diabetes:assessment_history')

# Admin dashboard view
@login_required
def admin_dashboard_view(request):
    """Admin dashboard to manage users based on role"""
    # Only allow staff users to access admin dashboard
    if not request.user.is_staff and request.user.role != 'DOCTOR':
        messages.error(request, 'You do not have permission to access the admin dashboard.')
        return redirect('diabetes:dashboard')
    
    # Get all users based on current user's role
    User = get_user_model()
    
    # Superadmin sees all users in hierarchical structure
    if request.user.role == 'SUPERADMIN':
        # Get all admins
        admin_users = User.objects.filter(role='ADMIN')
        
        # Get all doctors
        doctor_users = User.objects.filter(role='DOCTOR')
        
        # Get unassigned doctors (either no registered_by or registered by superadmin)
        unassigned_doctors = doctor_users.filter(
            models.Q(registered_by__isnull=True) |
            models.Q(registered_by__role='SUPERADMIN')
        )
        
        # Get all patients
        all_patients = Patient.objects.all().select_related('doctor')
        
        # Add patient counts to unassigned doctors
        for doctor in unassigned_doctors:
            doctor.patient_list = list(all_patients.filter(doctor=doctor))
            doctor.patient_count = len(doctor.patient_list)
        
        # Get doctors assigned to admins (exclude unassigned and superadmin-registered)
        assigned_doctors = doctor_users.filter(registered_by__role='ADMIN')
        
        # Create hierarchical structure
        for admin in admin_users:
            # Find doctors registered by this admin
            admin.managed_doctors = assigned_doctors.filter(registered_by=admin)
            admin.doctor_count = admin.managed_doctors.count()
            
            # For each doctor, find their patients
            for doctor in admin.managed_doctors:
                doctor.patient_list = list(all_patients.filter(doctor=doctor))
                doctor.patient_count = len(doctor.patient_list)
            
    # Admin sees only their registered doctors
    elif request.user.role == 'ADMIN':
        admin_users = User.objects.none()
        doctor_users = User.objects.filter(role='DOCTOR', registered_by=request.user)
        unassigned_doctors = User.objects.none()  # Admins don't see unassigned doctors
        
        # Get all patients for these doctors
        all_patients = Patient.objects.filter(doctor__in=doctor_users).select_related('doctor')
        
        # Add patient information to each doctor
        for doctor in doctor_users:
            doctor.patient_list = list(all_patients.filter(doctor=doctor))
            doctor.patient_count = len(doctor.patient_list)
            
    # Doctors shouldn't access this, but include empty querysets for safety
    else:
        admin_users = User.objects.none()
        doctor_users = User.objects.none()
        unassigned_doctors = User.objects.none()
    
    context = {
        'admin_users': admin_users,
        'doctor_users': doctor_users,
        'unassigned_doctors': unassigned_doctors,  # Always pass the queryset
        'user_role': request.user.role,
        'is_superadmin': request.user.role == 'SUPERADMIN',
        'is_admin': request.user.role == 'ADMIN',
    }
    
    return render(request, 'DiabetesPrediction/admin_dashboard.html', context)


# Doctor Reassignment View - allows superadmin to reassign doctors to different admins
@login_required
def reassign_doctor_view(request, doctor_id):
    """Reassign a doctor to a different admin"""
    # Only allow superadmin and admin users to access
    if request.user.role not in ['SUPERADMIN', 'ADMIN']:
        messages.error(request, 'You do not have permission to reassign doctors.')
        return redirect('diabetes:dashboard')
    
    # Get the doctor to reassign
    User = get_user_model()
    doctor = get_object_or_404(User, id=doctor_id, role='DOCTOR')
    
    # Check permissions: superadmin can reassign any doctor, admin can only reassign their own doctors
    if request.user.role == 'ADMIN' and doctor.registered_by != request.user:
        messages.error(request, 'You can only reassign doctors that you registered.')
        return redirect('diabetes:admin_dashboard')
    
    if request.method == 'POST':
        admin_id = request.POST.get('admin_id')
        
        # Handle 'null' option (unassign doctor from any admin)
        if admin_id == 'null':
            doctor.registered_by = None
            doctor.save()
            messages.success(request, f'Doctor {doctor.get_full_name()} has been unassigned from any administrator.')
        else:
            # Get the new admin
            try:
                new_admin = User.objects.get(id=admin_id, role='ADMIN')
                doctor.registered_by = new_admin
                doctor.save()
                messages.success(request, f'Doctor {doctor.get_full_name()} has been reassigned to administrator {new_admin.get_full_name()}.')
            except User.DoesNotExist:
                messages.error(request, 'The selected administrator does not exist.')
                
        return redirect('diabetes:admin_dashboard')
    else:
        # GET requests are not supported for this view
        messages.error(request, 'Invalid request method.')
        return redirect('diabetes:admin_dashboard')


# Patient Reassignment View - allows reassigning patients to different doctors
@login_required
def reassign_patient_view(request, patient_id):
    """Reassign a patient to a different doctor"""
    # Only allow superadmin and admin users to reassign patients
    if request.user.role not in ['SUPERADMIN', 'ADMIN']:
        messages.error(request, 'You do not have permission to reassign patients.')
        return redirect('diabetes:dashboard')
    
    # Get the patient to reassign
    patient = get_object_or_404(Patient, id=patient_id)
    
    # Check permissions: 
    # - Superadmin can reassign any patient
    # - Admin can only reassign patients of doctors they registered
    if request.user.role == 'ADMIN':
        if not patient.doctor.registered_by or patient.doctor.registered_by.id != request.user.id:
            messages.error(request, 'You can only reassign patients of doctors you manage.')
            return redirect('diabetes:admin_dashboard')
    
    if request.method == 'POST':
        doctor_id = request.POST.get('doctor_id')
        User = get_user_model()
        
        try:
            new_doctor = User.objects.get(id=doctor_id, role='DOCTOR')
            
            # Check if admin is trying to assign to a doctor they don't manage
            if request.user.role == 'ADMIN' and new_doctor.registered_by != request.user:
                messages.error(request, 'You can only reassign patients to doctors you manage.')
                return redirect('diabetes:admin_dashboard')
                
            # Update the patient's doctor
            old_doctor = patient.doctor
            patient.doctor = new_doctor
            patient.save()
            
            messages.success(request, 
                f'Patient {patient.full_name} has been reassigned from {old_doctor.get_full_name()} to {new_doctor.get_full_name()}.')
                
        except User.DoesNotExist:
            messages.error(request, 'The selected doctor does not exist.')
            
        return redirect('diabetes:admin_dashboard')
    else:
        # GET requests are not supported for this view
        messages.error(request, 'Invalid request method.')
        return redirect('diabetes:admin_dashboard')


@login_required
def patient_reassignment_view(request):
    # Only staff users (Superadmin and Admin) can reassign patients
    if not request.user.is_staff:
        messages.error(request, 'You do not have permission to reassign patients.')
        return redirect('diabetes:dashboard')
    
    # Get the doctor from the request if provided
    doctor_id = request.GET.get('doctor_id')
    doctor = None
    User = get_user_model()
    
    # Get doctors based on user role
    if request.user.role == 'SUPERADMIN':
        # Superadmin can see all doctors
        doctors = DiabetesUser.objects.filter(role='DOCTOR')
    else:
        # Admin can only see doctors they registered
        doctors = DiabetesUser.objects.filter(role='DOCTOR', registered_by=request.user)
    
    # Annotate doctors with patient count
    doctors = doctors.annotate(
        patient_count=Count('patients')
    ).order_by('first_name', 'last_name')
    
    if doctor_id:
        try:
            doctor = doctors.get(id=doctor_id)
            # Get patients for selected doctor
            patients = Patient.objects.filter(doctor=doctor).select_related('doctor')
        except User.DoesNotExist:
            messages.error(request, 'Doctor not found.')
            return redirect('diabetes:admin_dashboard')
    else:
        patients = None
    
    if request.method == 'POST':
        form = PatientReassignmentForm(request.POST, current_doctor=doctor, current_user=request.user)
        if form.is_valid():
            try:
                patient = form.cleaned_data['patient']
                new_doctor = form.cleaned_data['new_doctor']
                reason = form.cleaned_data['reason']
                
                if new_doctor == doctor:
                    messages.error(request, 'Cannot reassign patient to the same doctor.')
                else:
                    # Store information from the old patient record
                    old_doctor = patient.doctor
                    patient_name = patient.full_name
                    old_patient_id = patient.patient_id
                    
                    # Create a new patient record with the new doctor
                    new_patient = Patient(doctor=new_doctor, full_name=patient_name)
                    new_patient.save()
                    
                    # Update all assessments
                    DiabetesAssessment.objects.filter(patient=patient).update(
                        patient=new_patient,
                        doctor=new_doctor
                    )
                    
                    # Delete old patient record
                    patient.delete()
                    
                    messages.success(
                        request,
                        f'Patient {patient_name} (ID: {old_patient_id}) successfully reassigned from '
                        f'Dr. {old_doctor.get_full_name()} to Dr. {new_doctor.get_full_name()}. '
                        f'New ID: {new_patient.patient_id}'
                    )
                    return redirect('diabetes:admin_dashboard')
            except Exception as e:
                messages.error(request, f'Error during reassignment: {str(e)}')
    else:
        form = PatientReassignmentForm(current_doctor=doctor, current_user=request.user)
    
    context = {
        'form': form,
        'doctors': doctors,
        'selected_doctor': doctor,
        'patients': patients,
        'user_role': request.user.role,
    }
    
    return render(request, 'DiabetesPrediction/patient_reassignment.html', context)


# Delete user view
@login_required
def delete_admin(request, admin_id):
    """Delete an administrator and reassign their doctors"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    if not request.user.role == 'SUPERADMIN':
        return JsonResponse({'error': 'Permission denied'}, status=403)

    try:
        admin = get_object_or_404(DiabetesUser, id=admin_id, role='ADMIN')
        admin_name = admin.get_full_name()

        # Update doctors to be unassigned
        DiabetesUser.objects.filter(registered_by=admin).update(registered_by=None)

        # Delete the admin
        admin.delete()

        return JsonResponse({
            'success': True,
            'message': f'Administrator {admin_name} has been deleted successfully'
        })

    except DiabetesUser.DoesNotExist:
        return JsonResponse({'error': 'Administrator not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def delete_doctor(request, doctor_id):
    """Delete a doctor and handle their patients"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    if not (request.user.role in ['SUPERADMIN', 'ADMIN']):
        return JsonResponse({'error': 'Permission denied'}, status=403)

    try:
        doctor = get_object_or_404(DiabetesUser, id=doctor_id, role='DOCTOR')

        # If admin, check if they manage this doctor
        if request.user.role == 'ADMIN' and doctor.registered_by != request.user:
            return JsonResponse({'error': 'You can only delete doctors you manage'}, status=403)

        doctor_name = doctor.get_full_name()

        # Delete the doctor
        doctor.delete()

        return JsonResponse({
            'success': True,
            'message': f'Doctor {doctor_name} has been deleted successfully'
        })

    except DiabetesUser.DoesNotExist:
        return JsonResponse({'error': 'Doctor not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def delete_user(request, user_id):
    """Delete a user account based on role permissions"""
    # Only allow staff users to delete other users
    if not request.user.is_staff:
        messages.error(request, 'You do not have permission to delete user accounts.')
        return redirect('diabetes:dashboard')
    
    try:
        # Prevent self-deletion
        if int(user_id) == request.user.id:
            messages.error(request, 'You cannot delete your own account.')
            return redirect('diabetes:admin_dashboard')
            
        target_user = DiabetesUser.objects.get(id=user_id)
        target_email = target_user.email
        target_role = target_user.role
        
        # Check permission based on roles
        if request.user.role == 'SUPERADMIN':
            # Superadmin can delete both admins and doctors, but not other superadmins
            if target_role == 'SUPERADMIN':
                messages.error(request, 'Cannot delete another Superadmin account.')
                return redirect('diabetes:admin_dashboard')
                
            # Delete the user
            target_user.delete()
            if target_role == 'ADMIN':
                messages.success(request, f'Admin account for {target_email} has been successfully deleted.')
            else:
                messages.success(request, f'Doctor account for {target_email} has been successfully deleted.')
                
        elif request.user.role == 'ADMIN':
            # Admin can only delete doctors
            if target_role != 'DOCTOR':
                messages.error(request, 'You do not have permission to delete this account.')
                return redirect('diabetes:admin_dashboard')
                
            # Delete the doctor
            target_user.delete()
            messages.success(request, f'Doctor account for {target_email} has been successfully deleted.')
        else:
            # This shouldn't happen due to is_staff check, but handle it anyway
            messages.error(request, 'You do not have permission to delete accounts.')
            return redirect('diabetes:dashboard')
            
    except DiabetesUser.DoesNotExist:
        messages.error(request, 'User account not found.')
    except Exception as e:
        logger.error(f"Error in delete_user: {e}")
        messages.error(request, f"An error occurred: {str(e)}")
        
    return redirect('diabetes:admin_dashboard')

# Admin user registration view
@login_required
def admin_register_view(request):
    """View for registering new users based on current user's role"""
    # Only allow staff users to register new users
    if not request.user.is_staff:
        messages.error(request, 'You do not have permission to register new users.')
        return redirect('diabetes:dashboard')
    
    # Get user's role
    user_role = request.user.role
    
    # Determine what type of user can be registered based on current user's role
    if user_role == 'SUPERADMIN':
        register_type = request.GET.get('type', 'doctor').lower()
        
        # Superadmin can register both admins and doctors
        if register_type == 'admin':
            if request.method == 'POST':
                form = AdminRegistrationForm(request.POST)
                if form.is_valid():
                    user = form.save(commit=False)
                    user.username = user.email
                    user.save()
                    messages.success(request, f'Admin account for {user.email} created successfully!')
                    return redirect('diabetes:admin_dashboard')
            else:
                form = AdminRegistrationForm()
                
            context = {
                'form': form,
                'register_type': 'Admin',
                'title': 'Register New Administrator'
            }
            
        else:  # Default to doctor registration
            if request.method == 'POST':
                form = DoctorRegistrationForm(request.POST)
                if form.is_valid():
                    user = form.save(commit=False)
                    user.username = form.cleaned_data.get('email')
                    user.registered_by = request.user
                    user.save()
                    messages.success(request, f'Doctor account for {user.email} created successfully!')
                    return redirect('diabetes:admin_dashboard')
            else:
                form = DoctorRegistrationForm()
                
            context = {
                'form': form,
                'register_type': 'Doctor',
                'title': 'Register New Doctor'
            }
            
    elif user_role == 'ADMIN':
        # Admins can only register doctors
        if request.method == 'POST':
            form = DoctorRegistrationForm(request.POST)
            if form.is_valid():
                user = form.save(commit=False)
                user.registered_by = request.user
                user.save()
                messages.success(request, f'Doctor account for {user.email} created successfully!')
                return redirect('diabetes:admin_dashboard')
        else:
            form = DoctorRegistrationForm()
            
        context = {
            'form': form,
            'register_type': 'Doctor',
            'title': 'Register New Doctor'
        }
        
    else:
        # This shouldn't happen but handle it anyway
        messages.error(request, 'You do not have permission to register new users.')
        return redirect('diabetes:dashboard')
        
    return render(request, 'DiabetesPrediction/admin_register.html', context)

# User registration view
def register_view(request):
    """Register a new user"""
    if request.method == 'POST':
        form = DiabetesUserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, f'Account created for {user.email}! You can now login.')
            return redirect('diabetes:login')
    else:
        form = DiabetesUserRegistrationForm()
    
    return render(request, 'DiabetesPrediction/register.html', {'form': form})

# Superadmin registration view (should only be used for initial setup)
def superadmin_registration_view(request):
    """Register a new superadmin user - this should only be used for initial setup"""
    # Check if a superadmin already exists
    if DiabetesUser.objects.filter(role='SUPERADMIN').exists():
        messages.error(request, 'A Superadmin already exists. Only one Superadmin account is allowed.')
        return redirect('diabetes:login')
        
    if request.method == 'POST':
        form = SuperAdminRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, f'Superadmin account created for {user.email}! You can now login with superadmin privileges.')
            return redirect('diabetes:login')
    else:
        form = SuperAdminRegistrationForm()
    
    return render(request, 'DiabetesPrediction/superadmin_registration.html', {'form': form})

# Admin registration view - this is now protected and can only be accessed by a Superadmin
@login_required
def admin_registration_view(request):
    """Register a new admin user - only accessible by Superadmin"""
    # Check if user is a Superadmin
    if not request.user.is_staff or request.user.role != 'SUPERADMIN':
        messages.error(request, 'Only Superadmins can register new administrators.')
        return redirect('diabetes:dashboard')
        
    if request.method == 'POST':
        form = AdminRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, f'Admin account created for {user.email}! They can now login with admin privileges.')
            return redirect('diabetes:admin_dashboard')
    else:
        form = AdminRegistrationForm()
    
    return render(request, 'DiabetesPrediction/admin_registration.html', {'form': form})

# Training view for the admin
# train_model function removed during code cleanup - models are pre-trained

# User Profile Management
@login_required
def user_profile(request):
    """Display user profile with options to edit information"""
    return render(request, 'DiabetesPrediction/user_profile.html', {'user': request.user})

@login_required
def edit_profile(request):
    """Edit user profile information"""
    if request.method == 'POST':
        form = UserProfileUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile has been updated successfully.')
            return redirect('diabetes:user_profile')
    else:
        form = UserProfileUpdateForm(instance=request.user)
    
    return render(request, 'DiabetesPrediction/edit_profile.html', {'form': form})

@login_required
def change_password(request):
    """Change user password"""
    if request.method == 'POST':
        form = PasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            password = form.cleaned_data['new_password1']
            request.user.set_password(password)
            request.user.save()
            # Re-authenticate the user with the new password
            user = authenticate(username=request.user.email, password=password)
            login(request, user)
            messages.success(request, 'Your password has been changed successfully.')
            return redirect('diabetes:user_profile')
    else:
        form = PasswordChangeForm(user=request.user)
    
    return render(request, 'DiabetesPrediction/change_password.html', {'form': form})
