from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from DiabetesPrediction.models import Patient, DiabetesAssessment

class Command(BaseCommand):
    help = 'Deletes all users except for superadmins'

    def handle(self, *args, **options):
        User = get_user_model()
        
        # Get count of users before deletion
        total_users = User.objects.count()
        superadmins = User.objects.filter(role='SUPERADMIN').count()
        to_delete = total_users - superadmins
        
        # Delete all patients first (to avoid foreign key constraint issues)
        patient_count = Patient.objects.count()
        Patient.objects.all().delete()
        self.stdout.write(self.style.SUCCESS(f'Deleted {patient_count} patients'))
        
        # Delete all assessments associated with non-superadmin users
        assessment_count = DiabetesAssessment.objects.count()
        DiabetesAssessment.objects.all().delete()
        self.stdout.write(self.style.SUCCESS(f'Deleted {assessment_count} assessments'))
        
        # Delete all users except superadmins
        deleted = User.objects.exclude(role='SUPERADMIN').delete()[0]
        
        self.stdout.write(self.style.SUCCESS(
            f'Successfully deleted {deleted} users. {superadmins} superadmin(s) remain in the system.'
        ))
