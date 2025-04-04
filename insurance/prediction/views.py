from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import os
import joblib
import numpy as np
import pandas as pd

# Define paths relative to BASE_DIR
MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'insurance_model.pkl')

def home(request):
    """Render the home page template"""
    return render(request, 'home.html')

def predict(request):
    """Process prediction requests and return result as JSON"""
    if request.method == 'GET':
        try:
            # Fetch and validate query parameters
            parameters = ['Exposure', 'Area', 'VehPower', 'VehAge', 'DrivAge', 
                         'BonusMalus', 'VehBrand', 'VehGas', 'Density', 'Region']
            input_data = {}
            
            # Check if all parameters exist
            for param in parameters:
                value = request.GET.get(param)
                if value is None or value == '':
                    return JsonResponse({'error': f'Missing parameter: {param}'})
                input_data[param] = value
            
            # Load model (pipeline includes preprocessor)
            model = joblib.load(MODEL_PATH)
            
            # Create DataFrame with the input data
            features_df = pd.DataFrame([input_data])
            
            # Convert numeric fields to appropriate types
            numeric_cols = ['Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
            for col in numeric_cols:
                features_df[col] = pd.to_numeric(features_df[col])
            
            # Get prediction probability
            prediction_proba = model.predict_proba(features_df)[0][1]
            # Get binary prediction
            prediction = model.predict(features_df)[0]
            
            # Return both probability and binary prediction
            return JsonResponse({
                'claim_probability': round(float(prediction_proba), 4),
                'claim_predicted': int(prediction),
                'message': "Claim likely" if prediction == 1 else "No claim likely"
            })
            
        except ValueError as ve:
            return JsonResponse({'error': f'Invalid input format: {str(ve)}'})
        except Exception as e:
            return JsonResponse({'error': f'Prediction error: {str(e)}'})
    
    return JsonResponse({'error': 'Only GET requests are supported'})