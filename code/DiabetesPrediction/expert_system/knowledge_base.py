"""
Knowledge Base for Diabetes Expert System

This module contains medical knowledge, rules and facts about diabetes diagnosis and management.
"""

# Risk factors for diabetes
RISK_FACTORS = {
    'age': {
        'high': 45,  # Age 45 or older increases risk
        'weight': 2  # Importance weight for the factor
    },
    'bmi': {
        'normal': 18.5,
        'overweight': 25.0,
        'obese': 30.0,
        'weight': 3
    },
    'family_history': {
        'weight': 3  # Having family members with diabetes increases risk
    },
    'physical_activity': {
        'low': 150,  # Less than 150 minutes/week is considered low
        'weight': 2
    },
    'hypertension': {
        'weight': 2  # Presence of high blood pressure increases risk
    },
    'smoking': {
        'weight': 1
    }
}

# Clinical thresholds for diabetes diagnosis
DIAGNOSIS_THRESHOLDS = {
    'fasting_glucose': {
        'normal': 100,  # mg/dL
        'prediabetes': 100,  # 100-125 mg/dL range indicates prediabetes
        'diabetes': 126  # ≥126 mg/dL indicates diabetes
    },
    'glucose_tolerance': {
        'normal': 140,  # mg/dL
        'prediabetes': 140,  # 140-199 mg/dL range indicates prediabetes
        'diabetes': 200  # ≥200 mg/dL indicates diabetes
    },
    'hba1c': {
        'normal': 5.7,  # percent
        'prediabetes': 5.7,  # 5.7-6.4% range indicates prediabetes
        'diabetes': 6.5  # ≥6.5% indicates diabetes
    }
}

# Symptoms and their association with diabetes
SYMPTOMS = {
    'polyuria': {  # Frequent urination
        'weight': 3,
        'description': 'Excessive or abnormally large production of urine'
    },
    'polydipsia': {  # Excessive thirst
        'weight': 3,
        'description': 'Excessive thirst leading to drinking a lot of fluid'
    },
    'polyphagia': {  # Excessive hunger
        'weight': 2,
        'description': 'Excessive hunger or increased appetite'
    },
    'weight_loss': {
        'weight': 3,
        'description': 'Unexplained weight loss despite normal or increased eating'
    },
    'fatigue': {
        'weight': 2,
        'description': 'Feeling of tiredness, lack of energy'
    },
    'blurred_vision': {
        'weight': 2,
        'description': 'Inability to see fine details'
    },
    'slow_healing': {
        'weight': 2,
        'description': 'Cuts and wounds take longer to heal'
    },
    'tingling': {
        'weight': 2,
        'description': 'Tingling or numbness in hands or feet'
    }
}

# Diet recommendations based on diabetes status
DIET_RECOMMENDATIONS = {
    'prediabetes': [
        'Follow a balanced diet that limits added sugars and refined carbohydrates',
        'Focus on high-fiber foods (vegetables, fruits, legumes, whole grains)',
        'Include lean proteins with each meal (fish, poultry, tofu, legumes)',
        'Use the plate method: ½ non-starchy vegetables, ¼ lean protein, ¼ whole grains',
        'Choose healthy fats like olive oil, avocados, nuts, and seeds',
        'Practice portion control using measuring cups or a food scale',
        'Limit alcohol to 1 drink per day for women, 2 for men',
        'Consider consulting with a registered dietitian for a personalized meal plan'
    ],
    'diabetes': [
        'Track carbohydrate intake (aim for consistent amounts at each meal)',
        'Space meals evenly throughout the day to maintain stable blood glucose',
        'Choose high-fiber, low-glycemic index carbohydrates',
        'Include protein with each meal and snack to slow glucose absorption',
        'Limit saturated fats to less than 10% of daily calories',
        'Reduce sodium intake to less than 2,300mg per day',
        'Use the diabetes plate method: ½ non-starchy vegetables, ¼ lean protein, ¼ whole grains',
        'Avoid sugary beverages and choose water, unsweetened tea or coffee',
        'Read food labels and pay attention to serving sizes',
        'Consider medical nutrition therapy with a certified diabetes educator'
    ],
    'normal': [
        'Focus on a plant-forward eating pattern with abundant vegetables and fruits',
        'Choose whole grains over refined grains (brown rice, quinoa, whole wheat)',
        'Include healthy protein sources (fish, poultry, beans, nuts, low-fat dairy)',
        'Limit added sugars to less than 10% of daily calories',
        'Minimize ultra-processed foods and those high in sodium',
        'Stay hydrated primarily with water',
        'Follow the Mediterranean or DASH eating patterns',
        'Practice mindful eating by paying attention to hunger and fullness cues',
        'Maintain regular meal timing to support metabolic health'
    ],
    'low_risk': [
        'Maintain a balanced diet with plenty of vegetables, fruits, and whole grains',
        'Limit processed foods and added sugars',
        'Include lean proteins and healthy fats',
        'Stay hydrated with water as your primary beverage',
        'Be mindful of portion sizes even with healthy foods'
    ],
    'moderate_risk': [
        'Increase vegetable intake to at least 3-5 servings daily',
        'Choose whole grain carbohydrates and monitor portion sizes',
        'Include protein with meals to slow glucose absorption',
        'Reduce intake of refined carbohydrates and added sugars',
        'Limit alcohol consumption',
        'Consider keeping a food journal to identify patterns'
    ],
    'high_risk': [
        'Work with a dietitian to create a personalized meal plan',
        'Consider carbohydrate counting or the plate method',
        'Focus on fiber-rich foods that slow glucose absorption',
        'Minimize processed foods, sugary beverages, and desserts',
        'Space meals evenly throughout the day',
        'Monitor how different foods affect your energy levels',
        'Aim for a moderate weight loss of 5-7% if overweight'
    ]
}

# Physical activity recommendations
ACTIVITY_RECOMMENDATIONS = {
    'prediabetes': [
        'Aim for at least 150 minutes of moderate-intensity aerobic activity per week (30 minutes, 5 days/week)',
        'Break up exercise sessions into 10-15 minute segments if needed for better adherence',
        'Include resistance training 2-3 days per week targeting major muscle groups',
        'Reduce sedentary time by standing or walking for 3 minutes every 30 minutes',
        'Start with low-impact activities like walking, swimming, or cycling',
        'Use a step counter and gradually increase to 7,000-10,000 steps daily',
        'Schedule exercise at the same time each day to establish a routine',
        'Consider joining a diabetes prevention program with structured exercise guidance'
    ],
    'diabetes': [
        'Aim for 150-300 minutes of moderate-intensity activity weekly, spread across most days',
        'Include 2-3 sessions of resistance training targeting all major muscle groups',
        'Add flexibility and balance exercises 2-3 times weekly (yoga, tai chi, stretching)',
        'Monitor blood glucose before, during, and after exercise, especially if on insulin',
        'Carry fast-acting carbohydrates during exercise to treat potential hypoglycemia',
        'Avoid exercise during periods of very high blood glucose (>250 mg/dL with ketones)',
        'Wear proper footwear and moisture-wicking socks during all activities',
        'Stay hydrated and adjust insulin/medication as advised by healthcare provider',
        'Consider working with a certified diabetes educator or exercise physiologist'
    ],
    'normal': [
        'Engage in at least 150 minutes of moderate aerobic activity weekly',
        'Add 2 days of strength training for all major muscle groups',
        'Find enjoyable activities to ensure long-term adherence',
        'Break up sitting time with 2-3 minutes of movement every hour',
        'Consider using a fitness tracker to monitor daily activity levels',
        'Gradually increase intensity as fitness improves',
        'Incorporate incidental exercise into daily routines (taking stairs, walking meetings)',
        'Mix different types of activities for overall fitness and injury prevention'
    ],
    'low_risk': [
        'Maintain consistent physical activity of at least 150 minutes weekly',
        'Include both cardio and strength training in your routine',
        'Find activities you enjoy to promote long-term adherence',
        'Take regular movement breaks during prolonged sitting',
        'Gradually increase intensity as fitness improves'
    ],
    'moderate_risk': [
        'Aim for 150-225 minutes of moderate activity weekly',
        'Track your activity using a step counter or fitness app',
        'Schedule exercise sessions on your calendar to prioritize them',
        'Include both aerobic and resistance exercises',
        'Consider joining a group fitness class for accountability',
        'Break up sitting time with movement every hour'
    ],
    'high_risk': [
        'Work towards 225-300 minutes of physical activity weekly',
        'Start slowly and gradually increase duration and intensity',
        'Include both structured exercise and increased daily movement',
        'Consider working with a fitness professional to create a safe program',
        'Monitor how exercise affects your energy and glucose levels',
        'Include strength training 2-3 times weekly',
        'Choose low-impact activities if you have joint problems',
        'Celebrate small improvements in fitness and endurance'
    ]
}

# Lifestyle recommendations
LIFESTYLE_RECOMMENDATIONS = {
    'prediabetes': [
        'Aim for gradual weight loss of 5-7% if overweight (approximately 10-14 pounds for a 200-pound person)',
        'Quit smoking and avoid exposure to secondhand smoke',
        'Limit alcohol to moderate amounts (1 drink/day for women, 2 for men)',
        'Manage stress through mindfulness, meditation, yoga, or other relaxation techniques',
        'Get 7-9 hours of quality sleep each night',
        'Practice good sleep hygiene (consistent schedule, dark room, limit screens before bed)',
        'Stay current with recommended health screenings and vaccinations',
        'Consider using a continuous glucose monitor (CGM) if recommended by your doctor',
        'Join a structured diabetes prevention program (DPP)',
        'Share your prediabetes diagnosis with close family members to encourage their screening'
    ],
    'diabetes': [
        'Monitor blood glucose as recommended by your healthcare provider',
        'Learn to identify patterns in your glucose readings and what affects them',
        'Take medications exactly as prescribed and at consistent times',
        'Carry medical identification (bracelet, card) indicating you have diabetes',
        'Create a hypoglycemia action plan and share it with family/friends',
        'Learn to recognize and address symptoms of high and low blood sugar',
        'Maintain a consistent daily routine for meals, activity, and medication',
        'Attend all scheduled healthcare appointments (primary care, endocrinology, eye care, podiatry)',
        'Get recommended vaccinations (flu, pneumonia, hepatitis B, COVID-19)',
        'Consider joining a diabetes support group or working with a certified diabetes educator',
        'Take good care of your feet with daily inspections and proper footwear',
        'Practice good oral hygiene with regular dental check-ups',
        'Learn stress management techniques that work for you',
        'If you smoke, seek help to quit through your healthcare provider'
    ],
    'normal': [
        'Maintain a healthy weight with a BMI between 18.5-24.9',
        'Avoid all tobacco products and secondhand smoke',
        'Practice stress reduction techniques regularly',
        'Get regular health screenings based on your age and risk factors',
        'Prioritize quality sleep (7-9 hours) with consistent bedtime and wake schedules',
        'Drink alcohol in moderation if at all',
        'Stay socially connected with friends and family',
        'Engage in mentally stimulating activities',
        'Set up regular check-ups with your healthcare provider',
        'Know your family history of diabetes and other chronic conditions'
    ],
    'low_risk': [
        'Maintain a healthy weight through balanced nutrition and regular activity',
        'Schedule regular check-ups with your healthcare provider',
        'Know your numbers (blood pressure, cholesterol, blood sugar)',
        'Avoid tobacco products',
        'Prioritize quality sleep and stress management',
        'Consider annual screening if you have family history of diabetes'
    ],
    'moderate_risk': [
        'Aim for moderate weight loss of 5-7% if overweight',
        'Establish a consistent daily routine for meals and activity',
        'Get annual screenings for diabetes and related conditions',
        'Learn about early warning signs of diabetes',
        'Consider using a health tracking app for diet, exercise, and sleep',
        'Practice stress management techniques regularly',
        'Limit alcohol consumption',
        'Seek support from family and friends for healthy lifestyle changes'
    ],
    'high_risk': [
        'Work with healthcare providers to develop a comprehensive prevention plan',
        'Consider more frequent testing for diabetes (every 6 months)',
        'Track relevant health metrics (weight, blood pressure, glucose)',
        'If smoking, get help to quit through cessation programs',
        'Join a structured diabetes prevention program if available',
        'Set specific, measurable goals for lifestyle changes',
        'Create a consistent daily routine for meals, medication, and activity',
        'Learn stress management techniques that work for your lifestyle',
        'Ensure 7-9 hours of quality sleep each night',
        'Consider medication options for diabetes prevention if recommended'
    ]
}

# Complication risk factors and warning signs
COMPLICATIONS = {
    'cardiovascular': {
        'risk_factors': ['hypertension', 'high_cholesterol', 'smoking', 'obesity', 'family_history'],
        'warning_signs': ['chest_pain', 'shortness_of_breath', 'dizziness', 'irregular_heartbeat'],
        'recommendations': [
            'Monitor blood pressure and cholesterol regularly',
            'Take prescribed medications for blood pressure, cholesterol, and heart conditions',
            'Follow a heart-healthy diet low in sodium and unhealthy fats',
            'Stop smoking and avoid secondhand smoke',
            'Seek immediate medical attention for chest pain or discomfort'
        ]
    },
    'neuropathy': {
        'risk_factors': ['long_duration_diabetes', 'poor_glucose_control', 'smoking', 'alcohol_abuse'],
        'warning_signs': ['numbness', 'tingling', 'burning_sensation', 'loss_of_sensation', 'foot_ulcers'],
        'recommendations': [
            'Check feet daily for cuts, blisters, redness, swelling, or nail problems',
            'Wash feet daily and dry them carefully, especially between the toes',
            'Wear properly fitting shoes and socks',
            'Never walk barefoot, even indoors',
            'Have a foot exam at least once a year with your healthcare provider'
        ]
    },
    'nephropathy': {
        'risk_factors': ['hypertension', 'poor_glucose_control', 'smoking', 'family_history'],
        'warning_signs': ['swelling_in_legs', 'foamy_urine', 'fatigue', 'itchy_skin'],
        'recommendations': [
            'Control blood pressure and blood glucose',
            'Get urine and blood tests annually to check kidney function',
            'Follow a kidney-friendly diet if recommended',
            'Take ACE inhibitors or ARBs if prescribed',
            'Limit protein intake if advised by your healthcare provider'
        ]
    },
    'retinopathy': {
        'risk_factors': ['long_duration_diabetes', 'poor_glucose_control', 'hypertension', 'pregnancy'],
        'warning_signs': ['blurred_vision', 'floaters', 'vision_loss', 'eye_pain'],
        'recommendations': [
            'Get a comprehensive dilated eye exam at least once a year',
            'Control blood glucose, blood pressure, and cholesterol',
            'Stop smoking',
            'Promptly report any vision changes to your healthcare provider',
            'Consider protective eyewear for certain activities'
        ]
    }
}

# Medication information (simplified)
MEDICATIONS = {
    'metformin': {
        'class': 'Biguanide',
        'mechanism': 'Decreases glucose production in the liver and improves insulin sensitivity',
        'common_side_effects': ['Diarrhea', 'Nausea', 'Abdominal discomfort', 'Metallic taste'],
        'when_to_use': 'Usually first-line treatment for type 2 diabetes',
        'considerations': 'Take with food to reduce gastrointestinal side effects'
    },
    'sulfonylureas': {
        'class': 'Insulin secretagogue',
        'mechanism': 'Stimulates insulin release from the pancreas',
        'common_side_effects': ['Hypoglycemia', 'Weight gain', 'Skin reactions'],
        'when_to_use': 'When additional medication beyond metformin is needed',
        'considerations': 'Take with meals to reduce risk of hypoglycemia'
    },
    'dpp4_inhibitors': {
        'class': 'DPP-4 inhibitor',
        'mechanism': 'Increases incretin levels, which inhibit glucagon release and increase insulin secretion',
        'common_side_effects': ['Upper respiratory tract infection', 'Headache', 'Nasopharyngitis'],
        'when_to_use': 'Add-on therapy when metformin alone is inadequate',
        'considerations': 'Can be taken with or without food'
    },
    'sglt2_inhibitors': {
        'class': 'SGLT-2 inhibitor',
        'mechanism': 'Prevents reabsorption of glucose in the kidneys, increasing glucose excretion',
        'common_side_effects': ['Urinary tract infections', 'Genital yeast infections', 'Increased urination'],
        'when_to_use': 'Add-on therapy for type 2 diabetes with additional cardiovascular benefits',
        'considerations': 'Maintain adequate hydration and good genital hygiene'
    },
    'glp1_agonists': {
        'class': 'GLP-1 receptor agonist',
        'mechanism': 'Increases insulin secretion, decreases glucagon secretion, slows gastric emptying',
        'common_side_effects': ['Nausea', 'Vomiting', 'Diarrhea', 'Injection site reactions'],
        'when_to_use': 'Add-on therapy for type 2 diabetes, especially beneficial for weight loss',
        'considerations': 'Injectable medication; start with lower doses to minimize GI side effects'
    },
    'insulin': {
        'class': 'Insulin',
        'mechanism': 'Replaces or supplements the body\'s natural insulin',
        'common_side_effects': ['Hypoglycemia', 'Weight gain', 'Injection site reactions'],
        'when_to_use': 'Required for type 1 diabetes; used in type 2 when oral medications are inadequate',
        'considerations': 'Multiple types (rapid-acting, short-acting, intermediate-acting, long-acting)'
    }
}
