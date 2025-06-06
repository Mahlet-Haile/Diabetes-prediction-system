from django.core.management.base import BaseCommand
from django.db import connection

class Command(BaseCommand):
    help = 'Fix database by adding all required fields for the Diabetes Expert System'

    def handle(self, *args, **options):
        cursor = connection.cursor()
        
        # First, fix the NOT NULL constraint on heart_disease field
        self.stdout.write(self.style.SUCCESS('Fixing NOT NULL constraint on heart_disease field...'))
        try:
            # Create a temporary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temp_diabetesassessment AS 
                SELECT * FROM DiabetesPrediction_diabetesassessment;
            """)
            
            # Get the table schema
            cursor.execute("PRAGMA table_info(DiabetesPrediction_diabetesassessment)")
            columns = cursor.fetchall()
            
            # Create column definitions without NOT NULL on heart_disease
            column_defs = []
            for col in columns:
                col_name = col[1]  # Column name is the second element
                col_type = col[2]  # Column type is the third element
                
                # Special handling for id column (primary key)
                if col_name == 'id':
                    column_defs.append('id INTEGER PRIMARY KEY AUTOINCREMENT')
                    continue
                    
                # Handle all other columns
                # The safest approach is to make ALL fields nullable except id
                # This ensures the form will always work regardless of which fields are filled
                not_null = ''
                if col[3] and col_name == 'user_id':  # Only keep NOT NULL for user_id foreign key
                    not_null = 'NOT NULL'
                        
                default = f"DEFAULT {col[4]}" if col[4] is not None else ''
                column_defs.append(f"{col_name} {col_type} {not_null} {default}".strip())
            
            # Drop the original table
            cursor.execute("DROP TABLE DiabetesPrediction_diabetesassessment;")
            
            # Recreate the table with correct constraints
            cursor.execute(f"""
                CREATE TABLE DiabetesPrediction_diabetesassessment (
                    {', '.join(column_defs)}
                );
            """)
            
            # Copy data back
            cursor.execute("INSERT INTO DiabetesPrediction_diabetesassessment SELECT * FROM temp_diabetesassessment;")
            
            # Drop the temporary table
            cursor.execute("DROP TABLE temp_diabetesassessment;")
            
            self.stdout.write(self.style.SUCCESS('Successfully fixed NOT NULL constraint on heart_disease field'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error fixing heart_disease constraint: {str(e)}"))
        
        # List of fields we want to ensure exist in the database
        fields_to_add = [
            # Basic fields
            ("diabetes", "BOOLEAN NULL"),
            ("prediction_probability", "REAL NULL"),
            ("diabetes_type", "VARCHAR(50) NULL"),
            ("confidence_score", "REAL NULL"),
            
            # Physical measurements
            ("height", "REAL NULL"),
            ("weight", "REAL NULL"),
            ("cholesterol", "REAL NULL"),
            ("glucose", "REAL NULL"),
            ("fasting_glucose", "REAL NULL"),
            ("hba1c", "REAL NULL"),
            ("blood_pressure_systolic", "INTEGER NULL"),
            ("blood_pressure_diastolic", "INTEGER NULL"),
            
            # Lifestyle factors
            ("smoking", "BOOLEAN DEFAULT 0"),
            ("alcohol", "BOOLEAN DEFAULT 0"),
            ("active", "BOOLEAN DEFAULT 0"),
            ("physical_activity", "INTEGER NULL"),
            ("smoking_history", "VARCHAR(20) NULL"),
            ("family_history", "BOOLEAN DEFAULT 0"),
            
            # Symptoms
            ("polyuria", "BOOLEAN DEFAULT 0"),
            ("polydipsia", "BOOLEAN DEFAULT 0"), 
            ("polyphagia", "BOOLEAN DEFAULT 0"),
            ("weight_loss", "BOOLEAN DEFAULT 0"),
            ("fatigue", "BOOLEAN DEFAULT 0"),
            ("blurred_vision", "BOOLEAN DEFAULT 0"),
            ("slow_healing", "BOOLEAN DEFAULT 0"),
            ("tingling", "BOOLEAN DEFAULT 0"),
            
            # Complications
            ("chest_pain", "BOOLEAN DEFAULT 0"),
            ("shortness_of_breath", "BOOLEAN DEFAULT 0"),
            ("swelling_in_legs", "BOOLEAN DEFAULT 0"),
            ("numbness", "BOOLEAN DEFAULT 0"),
            ("foot_ulcers", "BOOLEAN DEFAULT 0"),
            ("vision_loss", "BOOLEAN DEFAULT 0"),
            
            # Expert system fields
            ("risk_score", "REAL NULL"),
            ("diagnosis", "VARCHAR(50) NULL"),
            ("expert_explanation", "TEXT NULL"),
            ("recommendations", "TEXT NULL"),
            ("complication_risks", "TEXT NULL"),
            
            # Recommendations
            ("diet_recommendations", "TEXT NULL"),
            ("exercise_recommendations", "TEXT NULL"),
            ("monitoring_recommendations", "TEXT NULL"),
            
            # Other symptom fields that might be in the form
            ("frequent_urination", "BOOLEAN DEFAULT 0"),
            ("excessive_thirst", "BOOLEAN DEFAULT 0"),
            ("unexplained_weight_loss", "BOOLEAN DEFAULT 0"),
            ("slow_healing_sores", "BOOLEAN DEFAULT 0"),
            ("numbness_tingling", "BOOLEAN DEFAULT 0"),
        ]
        
        success_count = 0
        error_count = 0
        
        self.stdout.write(self.style.SUCCESS('Starting database field repair...'))
        
        for field_name, field_type in fields_to_add:
            try:
                # Check if the column exists
                try:
                    cursor.execute(f"SELECT {field_name} FROM DiabetesPrediction_diabetesassessment LIMIT 1")
                    self.stdout.write(f"Column '{field_name}' already exists.")
                    success_count += 1
                except:
                    # Column doesn't exist, add it
                    cursor.execute(f"ALTER TABLE DiabetesPrediction_diabetesassessment ADD COLUMN {field_name} {field_type}")
                    self.stdout.write(self.style.SUCCESS(f"Added column '{field_name}' with type '{field_type}'"))
                    success_count += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error adding column '{field_name}': {str(e)}"))
                error_count += 1
        
        self.stdout.write(self.style.SUCCESS(f'Database repair complete! Successfully processed {success_count} fields with {error_count} errors.'))
