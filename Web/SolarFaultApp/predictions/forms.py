from django import forms

class FaultPredictionForm(forms.Form):
    ambient_temp = forms.FloatField(label="Ambient Temp (C)", required=True)
    solar_radiation = forms.FloatField(label="Solar Radiation (W/m2)", required=True)
    load_kw = forms.FloatField(label="Load kW Sum (kW)", required=True)
    inv_kw = forms.FloatField(label="Inv kW Sum (kW)", required=True)
    solar_kw = forms.FloatField(label="Solar kW (kW)", required=True)
