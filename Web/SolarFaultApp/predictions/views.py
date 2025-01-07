from django.shortcuts import render
from .forms import FaultPredictionForm
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('/Users/ansh/Music/AP/AP_Code/Major-Project/Web/solar_fault_model.pkl')

def predict_fault(request):
    prediction = None
    probabilities = None
    if request.method == 'POST':
        form = FaultPredictionForm(request.POST)
        if form.is_valid():
            # Prepare data for prediction
            input_data = pd.DataFrame([{
                'Ambient Temp (C)': form.cleaned_data['ambient_temp'],
                'Solar Radiation (W/m2)': form.cleaned_data['solar_radiation'],
                'Load kW Sum (kW)': form.cleaned_data['load_kw'],
                'Inv kW Sum (kW)': form.cleaned_data['inv_kw'],
                'Solar kW (kW)': form.cleaned_data['solar_kw']
            }])

            # Perform prediction
            probabilities = model.predict_proba(input_data)[0]
            prediction = model.classes_[probabilities.argmax()]

    else:
        form = FaultPredictionForm()

    context = {
        'form': form,
        'prediction': prediction,
        'probabilities': probabilities,
        'classes': model.classes_ if probabilities is not None else [],
    }
    return render(request, 'predictions/predict_fault.html', context)
