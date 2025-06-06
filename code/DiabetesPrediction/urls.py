from django.urls import path
from . import views
from . import nhanes_views

app_name = 'diabetes'

urlpatterns = [
    # Authentication URLs
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Application URLs
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('assessment/form/', views.assessment_form, name='assessment_form'),
    path('assessment/result/<int:assessment_id>/', views.assessment_result, name='assessment_result'),
    path('assessment/history/', views.assessment_history, name='assessment_history'),
    path('assessment/patient-data/', views.get_patient_data, name='get_patient_data'),
    path('assessment/delete/<int:assessment_id>/', views.delete_assessment, name='delete_assessment'),
    path('patient/delete/<int:patient_id>/', views.delete_patient, name='delete_patient'),
    # path('train/', views.train_model, name='train_model'),  # Training functionality disabled - models are pre-trained
    path('superadmin-register/', views.superadmin_registration_view, name='superadmin_registration'),
    path('admin-register/', views.admin_registration_view, name='admin_registration'),
    path('admin-dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('admin-register-user/', views.admin_register_view, name='admin_register'),
    path('delete-user/<int:user_id>/', views.delete_user, name='delete_user'),
    path('delete-admin/<int:admin_id>/', views.delete_admin, name='delete_admin'),
    path('delete-doctor/<int:doctor_id>/', views.delete_doctor, name='delete_doctor'),
    path('patient-reassignment/', views.patient_reassignment_view, name='patient_reassignment'),
    path('reassign-doctor/<int:doctor_id>/', views.reassign_doctor_view, name='reassign_doctor'),
    path('reassign-patient/<int:patient_id>/', views.reassign_patient_view, name='reassign_patient'),
    
    # NHANES assessment form (integrated into main system)
    path('nhanes/assessment/form/', nhanes_views.nhanes_assessment_form, name='nhanes_assessment_form'),
    
    # User Profile Management URLs
    path('profile/', views.user_profile, name='user_profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
    path('profile/change-password/', views.change_password, name='change_password'),
]
