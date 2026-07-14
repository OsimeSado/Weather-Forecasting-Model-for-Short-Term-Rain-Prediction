from django.urls import path
from . import views
urlpatterns = [
    path('', views.weather_view, name='weather_view'),
    path('api/locations/', views.location_suggestions, name='location_suggestions'),
]