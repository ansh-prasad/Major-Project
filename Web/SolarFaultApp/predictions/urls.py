from django.urls import path
from .views import predict_fault

urlpatterns = [
    path('', predict_fault, name='predict_fault'),
]
