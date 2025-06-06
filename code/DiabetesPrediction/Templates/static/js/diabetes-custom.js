/**
 * Custom JavaScript for Diabetes Prediction System
 */

document.addEventListener('DOMContentLoaded', function() {
    // Show tooltips for better user guidance
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // BMI Calculator functionality
    const heightInput = document.getElementById('height-input');
    const weightInput = document.getElementById('weight-input');
    const bmiResult = document.getElementById('bmi-result');
    const bmiInput = document.getElementById('id_bmi');
    
    function calculateBMI() {
        if (heightInput && weightInput && bmiResult && bmiInput) {
            const height = parseFloat(heightInput.value) / 100; // Convert cm to meters
            const weight = parseFloat(weightInput.value);
            
            if (height > 0 && weight > 0) {
                const bmi = weight / (height * height);
                bmiResult.textContent = bmi.toFixed(1);
                bmiInput.value = bmi.toFixed(1);
                
                // Update BMI category
                const bmiCategory = document.getElementById('bmi-category');
                if (bmiCategory) {
                    if (bmi < 18.5) {
                        bmiCategory.textContent = 'Underweight';
                        bmiCategory.className = 'text-info';
                    } else if (bmi < 25) {
                        bmiCategory.textContent = 'Normal';
                        bmiCategory.className = 'text-success';
                    } else if (bmi < 30) {
                        bmiCategory.textContent = 'Overweight';
                        bmiCategory.className = 'text-warning';
                    } else {
                        bmiCategory.textContent = 'Obese';
                        bmiCategory.className = 'text-danger';
                    }
                }
            }
        }
    }
    
    // Add event listeners for BMI calculator
    if (heightInput && weightInput) {
        heightInput.addEventListener('input', calculateBMI);
        weightInput.addEventListener('input', calculateBMI);
    }
    
    // Form validation
    const assessmentForm = document.getElementById('diabetes-assessment-form');
    if (assessmentForm) {
        assessmentForm.addEventListener('submit', function(event) {
            // Validate age
            const ageInput = document.getElementById('id_age');
            if (ageInput && (ageInput.value < 1 || ageInput.value > 120)) {
                event.preventDefault();
                alert('Please enter a valid age between 1 and 120.');
                ageInput.focus();
                return false;
            }
            
            // Validate clinical measurements if provided
            const glucoseInput = document.getElementById('id_blood_glucose_level');
            if (glucoseInput && glucoseInput.value !== '' && (glucoseInput.value < 50 || glucoseInput.value > 500)) {
                event.preventDefault();
                alert('Blood glucose should be between 50 and 500 mg/dL if provided.');
                glucoseInput.focus();
                return false;
            }
            
            const hba1cInput = document.getElementById('id_hba1c_level');
            if (hba1cInput && hba1cInput.value !== '' && (hba1cInput.value < 4 || hba1cInput.value > 15)) {
                event.preventDefault();
                alert('HbA1c should be between 4% and 15% if provided.');
                hba1cInput.focus();
                return false;
            }
        });
    }
    
    // Add symptom count indicator
    const symptomCheckboxes = document.querySelectorAll('.symptom-checkbox');
    const symptomCounter = document.getElementById('symptom-counter');
    
    function updateSymptomCount() {
        if (symptomCheckboxes && symptomCounter) {
            let count = 0;
            
            symptomCheckboxes.forEach(function(checkbox) {
                if (checkbox.checked) {
                    count++;
                }
            });
            
            symptomCounter.textContent = count;
        }
    }
    
    if (symptomCheckboxes) {
        symptomCheckboxes.forEach(function(checkbox) {
            checkbox.addEventListener('change', updateSymptomCount);
        });
        
        // Initial count
        updateSymptomCount();
    }
    
    // Risk level chart animation
    const riskCharts = document.querySelectorAll('.risk-chart');
    if (riskCharts) {
        riskCharts.forEach(function(chart) {
            const riskScore = parseFloat(chart.getAttribute('data-risk-score'));
            const riskIndicator = chart.querySelector('.risk-indicator');
            
            if (riskIndicator) {
                // Position the indicator based on risk score (0 to 1)
                const position = Math.min(Math.max(riskScore, 0), 1) * 100;
                riskIndicator.style.left = position + '%';
            }
        });
    }
});
