# prediction/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Default homepage
    path('predict/', views.predict, name='predict'),  # Predict route
]