# Skyline Weather App

> An intelligent Django-powered weather forecasting application that combines real-time weather APIs with Machine Learning to predict short-term rainfall using one year of historical weather data.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Django](https://img.shields.io/badge/Django-5.x-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![OpenWeatherMap](https://img.shields.io/badge/API-OpenWeatherMap-yellow)
![Open-Meteo](https://img.shields.io/badge/API-Open--Meteo-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## Overview

Skyline Weather App is a full-stack weather intelligence platform built with **Django**, **Machine Learning**, and **interactive data visualization**.

Unlike traditional weather applications that simply display weather information from an API, Skyline Weather App analyzes **365 days of hourly historical weather data**, engineers meaningful meteorological features, trains a **HistGradientBoosting Machine Learning model**, and predicts the probability of rainfall over the next six hours.

The application also provides weather insights, interactive forecast charts, historical weather trends, and automatically logs visitor activity through a custom Django middleware.

---

# Features

## Real-Time Weather

- Current weather conditions
- Feels-like temperature
- Minimum & maximum temperature
- Humidity
- Atmospheric pressure
- Cloud coverage
- Visibility
- Wind speed & compass direction
- Sunrise & sunset
- Day length

---

## Weather Forecasting

- 8-hour temperature forecast
- 8-hour humidity forecast
- 24-hour historical weather trends
- Interactive Chart.js visualizations
- Weather icons and dynamic UI

---

## Machine Learning Rain Prediction

Instead of relying solely on API forecasts, the application trains a custom Machine Learning model using historical weather observations.

### Model

- HistGradientBoostingClassifier
- TimeSeriesSplit Cross Validation
- Joblib Model Caching
- File Locking for concurrent requests

### Training Dataset

- 365 Days of hourly weather observations
- Open-Meteo Historical Archive API
- Automatic retraining every 24 hours

### Feature Engineering

The model learns from over **40 engineered weather features**, including:

- Temperature trends
- Humidity trends
- Pressure trends
- Rolling averages
- Rolling standard deviations
- Rainfall history
- Pressure fall detection
- Wind direction encoding
- Cyclic time encoding (hour/day)
- Seasonal indicators
- Lag features (1h, 3h, 6h)

The model predicts whether rain is likely within the next **6 hours** and outputs a probability score.

---

## Intelligent Weather Recommendations

Based on current weather conditions and ML predictions, the application generates contextual recommendations such as:

- Carry an umbrella
- Storm warnings
- Hydration reminders
- Cold weather advice
- High wind alerts
- Outdoor activity suggestions

---

## Interactive Charts

The dashboard includes:

- Temperature Forecast Chart
- Humidity Forecast Chart
- 24-Hour Historical Temperature Trend
- Historical Humidity Trend

Built with Chart.js.

---

## Visitor Tracking

Every visit is automatically logged using custom Django middleware.

Tracked information includes:

- Visitor IP Address
- Timestamp
- Requested Path
- City Searched

Logs are available through the Django Admin Dashboard.

---

# System Architecture

```
User
   │
   ▼
Django Web Application
   │
   ├───────────────► OpenWeatherMap API
   │
   ├───────────────► Open-Meteo Forecast API
   │
   ├───────────────► Open-Meteo Historical API
   │
   ▼
Feature Engineering
   │
   ▼
HistGradientBoosting Model
   │
   ▼
Rain Prediction
   │
   ▼
Recommendations + Charts
```

---

# Machine Learning Workflow

```
365 Days Historical Weather

        │

        ▼

Data Cleaning

        │

        ▼

Feature Engineering

        │

        ▼

Time Series Cross Validation

        │

        ▼

HistGradientBoosting Training

        │

        ▼

Model Caching (Joblib)

        │

        ▼

Real-Time Prediction
```

---

# Technologies Used

### Backend

- Django
- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Requests

### APIs

- OpenWeatherMap API
- Open-Meteo Forecast API
- Open-Meteo Historical Archive API

### Frontend

- HTML5
- CSS3
- JavaScript
- Chart.js

---

# Project Structure

```
Weather-App/
│
├── manage.py
├── requirements.txt
│
├── weatherProject/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── forecast/
    ├── admin.py
    ├── middleware.py
    ├── models.py
    ├── views.py
    ├── weather_trends.py
    ├── templates/
    │   └── weather.html
    └── static/
        ├── css/
        │   └── style.css
        └── js/
            ├── chartSetup.js
            └── trendChart.js
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/JeedyWhyte/Weather-App.git

cd Weather-App
```

## Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Migrations

```bash
python manage.py migrate
```

---

## Configure OpenWeatherMap API (Optional)

Windows

```powershell
$env:OWM_API_KEY="your_api_key"
```

Linux/macOS

```bash
export OWM_API_KEY="your_api_key"
```

If no API key is supplied, the application uses a development key.

---

# Run the Application

```bash
python manage.py runserver
```

Open:

```
http://127.0.0.1:8000/
```

---

# Django Admin

Create an administrator account

```bash
python manage.py createsuperuser
```

Login at

```
http://127.0.0.1:8000/admin/
```

Visitor logs are available under **Site Visitors**.

---

# Future Improvements

- Docker deployment
- User authentication
- Weather alerts & notifications
- Multi-city dashboard
- Model explainability (SHAP)
- Live radar integration
- Weekly rainfall prediction
- CI/CD with GitHub Actions
- Cloud deployment (Render/Railway)

---

# Author

### Sado Osimeozemeokhai
Computer Engineer • Machine Learning Enthusiast • Software Developer

**GitHub Repository:** https://github.com/OsimeSado

### Bryan Aghogho
• Software Developer

**GitHub Repository:** https://github.com/JeedyWhyte

