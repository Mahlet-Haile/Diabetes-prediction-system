from django.core.management.base import BaseCommand
from django.db import transaction
from DiabetesPrediction.models import DiabetesUser, Patient, DiabetesAssessment

class Command(BaseCommand):
    help = 'Reset all users and keep only the superadmin account'

    def handle(self, *args, **options):
        with transaction.atomic():
            # Delete all assessments
            assessment_count = DiabetesAssessment.objects.all().count()
            DiabetesAssessment.objects.all().delete()
            self.stdout.write(self.style.SUCCESS(f'Deleted {assessment_count} assessments'))
            
            # Delete all patients
            patient_count = Patient.objects.all().count()
            Patient.objects.all().delete()
            self.stdout.write(self.style.SUCCESS(f'Deleted {patient_count} patients'))
            
            # Get current user count
            user_count = DiabetesUser.objects.count()
            
            # Delete all users
            DiabetesUser.objects.all().delete()
            self.stdout.write(self.style.SUCCESS(f'Deleted {user_count} users'))
            
            # Create the superadmin user
            superadmin = DiabetesUser.objects.create_superuser(
                email='mahlethaile@gmail.com',
                username='mahlethaile',
                password='Mahi@12341234',
                first_name='Mahlet',
                last_name='Haile',
                role='SUPERADMIN'
            )
            self.stdout.write(self.style.SUCCESS(f'Created superadmin user: {superadmin.email}'))
            
        self.stdout.write(self.style.SUCCESS('Reset completed successfully!'))
