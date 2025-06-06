# Diabetes Prediction System

A comprehensive web-based system for diabetes risk assessment and prediction using machine learning models and expert systems.

## Features

- **Multiple Assessment Methods**
  - Basic diabetes risk assessment
  - Comprehensive health assessment
  - NHANES-based assessment
  - Expert system analysis

- **User Management**
  - Patient registration and profiles
  - Doctor/Admin dashboards
  - Role-based access control
  - Secure authentication

- **Advanced Analytics**
  - Machine learning-based predictions
  - Risk factor identification
  - Historical assessment tracking

## Technical Stack

- **Backend**: Django 4.0.4
- **Database**: MySQL
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: XGBoost, Scikit-learn
- **Security**: Encrypted data storage, CSRF protection

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Mahlet-Haile/Diabetes-prediction-system.git
   cd Diabetes-prediction-system
   ```

2. **Set Up Database**
   - Install MySQL
   - Run the setup script:
     ```bash
     mysql -u root -p < code/mysql_setup.sql
     ```

3. **Install Dependencies**
   ```bash
   pip install django==4.0.4
   pip install mysqlclient
   pip install python-decouple
   pip install cryptography
   pip install crispy-forms
   ```

4. **Configure Environment**
   Create a `.env` file in the `code` directory:
   ```
   DJANGO_SECRET_KEY=your-secret-key
   DB_NAME=diabetes_prediction
   DB_USER=diabetes_user
   DB_PASSWORD=diabetes_password
   DB_HOST=localhost
   DB_PORT=3306
   ```

5. **Run Migrations**
   ```bash
   cd code
   python manage.py migrate
   python manage.py createsuperuser
   ```

6. **Start the Server**
   ```bash
   python manage.py runserver
   ```

## Project Structure

```
Diabetes-prediction-system/
├── code/
│   ├── DiabetesPrediction/        # Main application
│   │   ├── Templates/            # HTML templates
│   │   ├── expert_system/        # Expert system implementation
│   │   ├── migrations/           # Database migrations
│   │   └── nhanes_data/         # NHANES dataset integration
│   ├── FinalProject/            # Django project settings
│   └── manage.py                # Django management script
└── README.md
```

## Usage

1. **Access the Application**
   - Open your browser and go to `http://localhost:8000`
   - log in with existing credentials superadmin "mahlethaile@gmail.com" passwword "Mahi@12341234"

2. **User Roles**
   - **Patients**: Can take assessments and view their results
   - **Doctors**: Can manage patients and view assessment results
   - **Admins**: Can manage users and system settings

3. **Taking Assessments**
   - Choose the type of assessment
   - Fill in the required health information
   - Submit to get your risk assessment

## Security Features

- Encrypted sensitive data
- CSRF protection
- Secure session management
- Role-based access control
- Password validation and hashing

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Mahlet Haile - [GitHub Profile](https://github.com/Mahlet-Haile)

Project Link: [https://github.com/Mahlet-Haile/Diabetes-prediction-system](https://github.com/Mahlet-Haile/Diabetes-prediction-system) 