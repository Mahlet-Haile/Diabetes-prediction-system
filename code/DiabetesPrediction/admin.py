from django.contrib import admin
from .models import DiabetesAssessment, DiabetesUser, Patient

class PatientAdmin(admin.ModelAdmin):
    list_display = ('patient_id', 'full_name', 'doctor', 'date_added')
    list_filter = ('doctor', 'date_added')
    search_fields = ('full_name', 'patient_id', 'doctor__username')
    ordering = ('doctor', 'patient_id')

class DiabetesAssessmentAdmin(admin.ModelAdmin):
    list_display = ('patient', 'doctor', 'assessment_date', 'diagnosis', 'risk_score')
    list_filter = ('diagnosis', 'assessment_date', 'doctor')
    search_fields = ('patient__full_name', 'patient__patient_id', 'doctor__username', 'diagnosis')
    ordering = ('-assessment_date',)
    
    readonly_fields = ('recommendations', 'complication_risks', 'expert_explanation')

admin.site.register(DiabetesAssessment, DiabetesAssessmentAdmin)
admin.site.register(Patient, PatientAdmin)
