MVP Link - https://motor-insurance.onrender.com/

# Insurance Claim Prediction API

A Django-based web application that predicts the likelihood of insurance claims based on policy and vehicle information. This project implements a binary classification model that determines whether a policy is likely to have a claim.

## Overview

This application uses machine learning to predict whether an insurance policy will result in a claim. It takes various policy and vehicle details as input and returns a prediction probability along with a binary outcome (claim/no claim).

## Features

- Binary classification model to predict insurance claims
- REST API endpoint for making predictions
- Simple web interface for testing predictions
- Model training script with preprocessing pipeline

## Project Structure

```
insurance-claim-predictor/
│
├── models/                      # Directory to store ML models
│   ├── insurance_model.pkl      # Trained machine learning model
│
├── prediction/                  # Django app
│   ├── __init__.py
│   ├── models.py                # Database models
│   ├── views.py                 # API endpoints and view functions
│   ├── urls.py                  # URL routing
│   ├── templates/               # HTML templates
│   │   └── home.html            # Main prediction interface
│
├── train_model.py               # Script to train and save the model
├── insurance_data.csv           # Dataset (not included in repository)
├── manage.py                    # Django management script
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/insurance-claim-predictor.git
   cd insurance-claim-predictor
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Train the model:
   ```
   python train_model.py
   ```
   Note: You'll need to place your `insurance_data.csv` file in the project root.

5. Run migrations:
   ```
   python manage.py makemigrations
   python manage.py migrate
   ```

6. Start the development server:
   ```
   python manage.py runserver
   ```

7. Access the application at http://127.0.0.1:8000/

## Dataset

The model uses the following features:
- IDpol: Policy ID
- ClaimNb: Number of claims (target variable)
- Exposure: Exposure time
- Area: Geographic area type
- VehPower: Vehicle power
- VehAge: Vehicle age in years
- DrivAge: Driver age in years
- BonusMalus: Bonus-malus score
- VehBrand: Vehicle brand/manufacturer
- VehGas: Vehicle fuel type
- Density: Population density
- Region: Geographic region

## Model Training

The model is a RandomForest classifier with preprocessing for both numerical and categorical features. To train the model with a reduced dataset size (for quicker iterations):

```python
# Modify the SAMPLE_FRACTION in train_model.py
SAMPLE_FRACTION = 0.1  # Uses 10% of the data
```

## API Usage

### Endpoint: `/predict/`

**Method:** GET

**Parameters:**
- Exposure: Float
- Area: String
- VehPower: Integer
- VehAge: Integer
- DrivAge: Integer
- BonusMalus: Integer
- VehBrand: String
- VehGas: String
- Density: Float
- Region: String

**Response:**
```json
{
  "claim_probability": 0.35,
  "claim_predicted": 0,
  "message": "No claim likely"
}
```

## Performance Considerations

- With 678,000 entries in the full dataset, training can take several hours on standard hardware
- Consider using a subset of the data for initial development
- The final model should be trained on the complete dataset for production use

## Requirements

- Python 3.8+
- Django 4.0+
- scikit-learn 1.0+
- pandas
- numpy
- joblib

## License

[MIT License](LICENSE)

## Future Improvements

- Add user authentication
- Implement model versioning
- Add feature importance visualization
- Include more advanced models (e.g., XGBoost, neural networks)
- Create a batch prediction interface
