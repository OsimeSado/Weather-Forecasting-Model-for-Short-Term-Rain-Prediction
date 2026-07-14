# Skyline Weather App

A Django weather application with ML-powered rain prediction, interactive charts, and visitor tracking.

## Features

- Real-time weather data from OpenWeatherMap and Open-Meteo APIs
- 8-hour temperature and humidity forecast chart
- 24-hour historical trend chart
- ML rain prediction using HistGradientBoosting trained on 365 days of hourly data
- Weather-based recommendations
- Visitor IP address tracking

## Requirements

- Python 3.10+
- An OpenWeatherMap API key (a default is included for development)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/JeedyWhyte/Weather-App.git
   cd Weather-App
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run database migrations:**

   ```bash
   python manage.py migrate
   ```

5. **(Optional) Set your OpenWeatherMap API key:**

   ```bash
   # Windows PowerShell
   $env:OWM_API_KEY = "your_api_key_here"

   # macOS / Linux
   export OWM_API_KEY="your_api_key_here"
   ```

   If not set, the app uses a built-in development key.

## Running the App

```bash
python manage.py runserver
```

Open your browser and go to `http://127.0.0.1:8000/`.

## Viewing Tracked Visitors

Visitor IP addresses are stored automatically on every page visit. To view them:

1. Create a superuser:

   ```bash
   python manage.py createsuperuser
   ```

2. Go to `http://127.0.0.1:8000/admin/` and log in.

3. Click **Site visitors** to see all tracked IP addresses with timestamps, paths, and cities searched.

## Project Structure

```
Weather App/
├── manage.py
├── requirements.txt
├── weatherProject/          # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── forecast/                # Main app
    ├── models.py            # SiteVisitor model for IP tracking
    ├── views.py             # Weather logic, API calls, ML model
    ├── middleware.py         # IP tracking middleware
    ├── weather_trends.py    # 24-hour trend data helper
    ├── admin.py             # Admin registration for visitor logs
    ├── static/
    │   ├── css/style.css
    │   └── js/
    │       ├── chartSetup.js
    │       └── trendChart.js
    └── templates/
        └── weather.html
```
