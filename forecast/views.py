from django.shortcuts import render
from django.http import JsonResponse

import os
import requests
import pandas as pd
import numpy as np
import joblib

from datetime import datetime, timedelta, timezone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score

from .weather_trends import get_trend_data

try:
    from filelock import FileLock
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False
    print("Warning: filelock module not found. Install with pip install filelock")


OPENWEATHERMAP_API_KEY = os.getenv("OWM_API_KEY", "decb14ec0acc76cbccae785f023d00c9")
OPENWEATHERMAP_BASE_URL = "https://api.openweathermap.org/data/2.5/"

MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# 1. FETCH CURRENT WEATHER FROM OPENWEATHERMAP
def get_current_weather(city):
    url = f"{OPENWEATHERMAP_BASE_URL}weather?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"

    response = requests.get(url, timeout=10)
    data = response.json()

    if response.status_code != 200:
        raise Exception(data.get("message", "City not found"))

    return _parse_weather_response(data)


def get_current_weather_by_coords(lat, lon):
    url = f"{OPENWEATHERMAP_BASE_URL}weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"

    response = requests.get(url, timeout=10)
    data = response.json()

    if response.status_code != 200:
        raise Exception(data.get("message", "Location not found"))

    return _parse_weather_response(data)


def _parse_weather_response(data):
    timezone_offset = data.get("timezone", 0)

    sunrise_utc = datetime.fromtimestamp(data["sys"]["sunrise"], tz=timezone.utc)
    sunset_utc = datetime.fromtimestamp(data["sys"]["sunset"], tz=timezone.utc)

    local_tz = timezone(timedelta(seconds=timezone_offset))
    sunrise_local = sunrise_utc.astimezone(local_tz)
    sunset_local = sunset_utc.astimezone(local_tz)

    day_length = (sunset_local - sunrise_local).seconds / 3600

    return {
        "city": data["name"],
        "country": data["sys"]["country"],
        "current_temp": round(data["main"]["temp"], 1),
        "feels_like": round(data["main"]["feels_like"], 1),
        "temp_min": round(data["main"]["temp_min"], 1),
        "temp_max": round(data["main"]["temp_max"], 1),
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "description": data["weather"][0]["description"],
        "icon": data["weather"][0]["icon"],
        "main_weather": data["weather"][0]["main"],
        "clouds": data["clouds"]["all"],
        "visibility": data.get("visibility", 10000),
        "wind_speed": round(data["wind"]["speed"] * 3.6, 1),
        "wind_deg": data["wind"].get("deg", 0),
        "lat": data["coord"]["lat"],
        "lon": data["coord"]["lon"],
        "sunrise": sunrise_local.strftime("%H:%M"),
        "sunset": sunset_local.strftime("%H:%M"),
        "day_length": round(day_length, 1),
        "timezone_offset": timezone_offset,
    }


# 1b. REVERSE GEOCODING
def reverse_geocode(lat, lon):
    url = "https://api.openweathermap.org/geo/1.0/reverse"
    params = {"lat": lat, "lon": lon, "limit": 1, "appid": OPENWEATHERMAP_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data and len(data) > 0:
            item = data[0]
            name = item.get("name", "Unknown")
            state = item.get("state", "")
            country = item.get("country", "")
            parts = [p for p in [name, state, country] if p]
            return {"name": name, "state": state, "country": country, "display_name": ", ".join(parts)}
    except Exception as e:
        print(f"Reverse geocode error: {e}")
    return {"name": "Unknown", "state": "", "country": "", "display_name": f"{lat}, {lon}"}


# 1c. LOCATION SEARCH SUGGESTIONS
BROAD_LOCATION_TYPES = {"country"}

def location_suggestions(request):
    query = request.GET.get("q", "").strip()
    if len(query) < 2:
        return JsonResponse([], safe=False)

    url = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": query, "limit": 5, "appid": OPENWEATHERMAP_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        results = []
        for item in data:
            name = item.get("name", "")
            state = item.get("state", "")
            country = item.get("country", "")
            lat = item.get("lat")
            lon = item.get("lon")
            parts = [p for p in [name, state, country] if p]
            display_name = ", ".join(parts)

            is_broad = (
                not state
                and name.lower() == query.lower()
                and len(data) == 1
                and country
                and name.lower() == country.lower()
            )

            results.append({
                "name": name,
                "state": state,
                "country": country,
                "lat": lat,
                "lon": lon,
                "display_name": display_name,
                "is_broad": is_broad,
            })

        return JsonResponse(results, safe=False)
    except Exception as e:
        print(f"Location suggestions error: {e}")
        return JsonResponse([], safe=False)


# 2. FETCH RECENT WEATHER AND FORECAST FROM OPEN METEO
def fetch_forecast_and_history(lat, lon, hours_forcast=12):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "precipitation",
            "precipitation_probability",
            "rain",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "past_days": 1,
        "forecast_days": 2,
        "timezone": "auto",
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "temperature_2m": data["hourly"]["temperature_2m"],
            "relative_humidity_2m": data["hourly"]["relative_humidity_2m"],
            "surface_pressure": data["hourly"]["surface_pressure"],
            "precipitation": data["hourly"]["precipitation"],
            "precipitation_probability": data["hourly"]["precipitation_probability"],
            "rain": data["hourly"]["rain"],
            "cloud_cover": data["hourly"]["cloud_cover"],
            "wind_speed_10m": data["hourly"]["wind_speed_10m"],
            "wind_direction_10m": data["hourly"]["wind_direction_10m"],
        })

        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        df["time_naive"] = df["time"].dt.tz_localize(None)
        current_idx = (df["time_naive"] - current_hour).abs().idxmin()

        return {
            "df": df,
            "current_idx": current_idx,
            "timezone": data.get("timezone", "UTC"),
        }
    except Exception as e:
        print(f"Error fetching forecast and history: {e}")
        return None


# 3. FETCH HOURLY HISTORICAL WEATHER DATA FROM OPEN-METEO (FOR TRAINING)
def fetch_historical_data(lat, lon, days_back=365):
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "precipitation",
            "rain",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
        ],
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        data = response.json()

        if 'hourly' not in data:
            print(f"Historical data error: {data}")
            return None

        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

        df = df.asfreq('h')
        df = df.ffill(limit=6)
        df = df.dropna()
        df = df.reset_index()

        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None


# 4. FEATURE ENGINEERING
def engineer_features(df):
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    df["rain_next_6h"] = (
        df["rain"].rolling(window=6, min_periods=1).sum().shift(-6) > 0.1
    ).astype(int)

    df["hour"] = df["time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_year"] = df["time"].dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["month"] = df["time"].dt.month
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

    df["temp_1h_ago"] = df["temperature_2m"].shift(1)
    df["temp_3h_ago"] = df["temperature_2m"].shift(3)
    df["temp_6h_ago"] = df["temperature_2m"].shift(6)
    df["temp_change_1h"] = df["temperature_2m"] - df["temp_1h_ago"]
    df["temp_change_3h"] = df["temperature_2m"] - df["temp_3h_ago"]
    df["temp_change_6h"] = df["temperature_2m"] - df["temp_6h_ago"]

    df["humidity_1h_ago"] = df["relative_humidity_2m"].shift(1)
    df["humidity_change_1h"] = df["relative_humidity_2m"] - df["humidity_1h_ago"]

    df["pressure_1h_ago"] = df["surface_pressure"].shift(1)
    df["pressure_3h_ago"] = df["surface_pressure"].shift(3)
    df["pressure_6h_ago"] = df["surface_pressure"].shift(6)
    df["pressure_change_1h"] = df["surface_pressure"] - df["pressure_1h_ago"]
    df["pressure_change_3h"] = df["surface_pressure"] - df["pressure_3h_ago"]
    df["pressure_change_6h"] = df["surface_pressure"] - df["pressure_6h_ago"]

    df["temp_6h_avg"] = df["temperature_2m"].rolling(6).mean().shift(1)
    df["temp_6h_std"] = df["temperature_2m"].rolling(6).std().shift(1)
    df["humidity_6h_avg"] = df["relative_humidity_2m"].rolling(6).mean().shift(1)
    df["pressure_6h_avg"] = df["surface_pressure"].rolling(6).mean().shift(1)
    df["rain_6h_sum"] = df["rain"].rolling(6).sum().shift(1)
    df["rain_24h_sum"] = df["rain"].rolling(24).sum().shift(1)

    df["rain_last_hour"] = (df["rain"].shift(1) > 0).astype(int)
    df["rain_last_3h"] = (df["rain"].rolling(3).sum().shift(1) > 0).astype(int)
    df["rain_last_6h"] = (df["rain"].rolling(6).sum().shift(1) > 0).astype(int)

    df["wind_dir_sin"] = np.sin(np.radians(df["wind_direction_10m"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["wind_direction_10m"]))

    df["pressure_falling"] = (df["pressure_change_3h"] < -1).astype(int)
    df["pressure_rising"] = (df["pressure_change_3h"] > 1).astype(int)

    df = df.dropna()

    return df


# 5. PREPARE TRAINING DATA
FEATURE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_dir_sin", "wind_dir_cos", "cloud_cover",

    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "month", "is_winter", "is_summer",

    "temp_1h_ago", "temp_3h_ago", "temp_6h_ago",
    "temp_change_1h", "temp_change_3h", "temp_change_6h",

    "humidity_1h_ago", "humidity_change_1h",

    "pressure_1h_ago", "pressure_3h_ago", "pressure_6h_ago",
    "pressure_change_1h", "pressure_change_3h", "pressure_change_6h",

    "temp_6h_avg", "temp_6h_std", "humidity_6h_avg", "pressure_6h_avg",

    "rain_6h_sum", "rain_24h_sum",
    "rain_last_hour", "rain_last_3h", "rain_last_6h",

    "pressure_falling", "pressure_rising"
]


def prepare_training_data(df):
    X = df[FEATURE_COLUMNS]
    y = df["rain_next_6h"]
    return X, y


# 6. TRAIN THE RAIN PREDICTION MODEL
def train_rain_model(X, y):
    unique_classes = y.nunique()

    if unique_classes < 2:
        print(f"Warning: Only {unique_classes} class. Using fallback model.")

        class DummyModel:
            def __init__(self, default):
                self.default = default

            def predict(self, X):
                return np.array([self.default] * len(X))

            def predict_proba(self, X):
                if self.default == 0:
                    return np.array([[0.85, 0.15]] * len(X))
                return np.array([[0.15, 0.85]] * len(X))

        return DummyModel(y.iloc[0] if len(y) > 0 else 0), {'f1': 0.5, 'accuracy': 0.5}

    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
        class_weight="balanced"
    )

    tscv = TimeSeriesSplit(n_splits=5)

    f1_scores = []
    acc_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_train.nunique() < 2:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        acc_scores.append(accuracy_score(y_test, y_pred))

    model.fit(X, y)

    metrics = {
        'f1': np.mean(f1_scores) if f1_scores else 0.5,
        'accuracy': np.mean(acc_scores) if acc_scores else 0.5
    }
    print(f"Rain Model - F1: {metrics['f1']:.2%}, Accuracy: {metrics['accuracy']:.2%}")

    return model, metrics


# 7. MODEL CACHING WITH FILE LOCKING
def get_or_train_model(lat, lon):
    cache_key = f"{round(lat, 1)}_{round(lon, 1)}"
    model_path = os.path.join(MODEL_DIR, f"rain_model_{cache_key}.joblib")
    lock_path = os.path.join(MODEL_DIR, f"rain_model_{cache_key}.lock")

    def load_model():
        if os.path.exists(model_path):
            age = datetime.now().timestamp() - os.path.getmtime(model_path)
            if age < 86400:
                print("\nLoading saved model...")
                return joblib.load(model_path)
        return None

    def train_and_save_model():
        print("\nFetching 365 days historical weather data...")
        df = fetch_historical_data(lat, lon, days_back=365)

        if df is None or len(df) < 100:
            print("Not enough historical data to train model.")
            return None

        print(f"Engineering features from {len(df)} hourly records...")
        df = engineer_features(df)
        if len(df) < 100:
            print("Not enough data after feature engineering to train model.")
            return None

        print("Training rain prediction model with HistGradientBoosting...")
        X, y = prepare_training_data(df)
        model, metrics = train_rain_model(X, y)

        joblib.dump(model, model_path)
        print("Model saved successfully.")

        return model

    model = load_model()
    if model is not None:
        return model

    if HAS_FILELOCK:
        with FileLock(lock_path, timeout=180):
            model = load_model()
            if model is not None:
                return model
            return train_and_save_model()
    else:
        return train_and_save_model()


# 8. PREDICTION FUNCTION (WITH REAL-TIME DATA)
def predict_rain_probability(model, current_weather, forecast_data):
    if forecast_data is None:
        return 0.3, 0

    df = forecast_data["df"]
    idx = forecast_data["current_idx"]

    if idx < 6:
        return 0.3, 0

    now = datetime.now()

    temp_now = current_weather["current_temp"]
    temp_1h = df.loc[idx - 1, "temperature_2m"] if idx >= 1 else temp_now
    temp_3h = df.loc[idx - 3, "temperature_2m"] if idx >= 3 else temp_now
    temp_6h = df.loc[idx - 6, "temperature_2m"] if idx >= 6 else temp_now

    humidity_now = current_weather["humidity"]
    humidity_1h = df.loc[idx - 1, "relative_humidity_2m"] if idx >= 1 else humidity_now

    pressure_now = current_weather["pressure"]
    pressure_1h = df.loc[idx - 1, "surface_pressure"] if idx >= 1 else pressure_now
    pressure_3h = df.loc[idx - 3, "surface_pressure"] if idx >= 3 else pressure_now
    pressure_6h = df.loc[idx - 6, "surface_pressure"] if idx >= 6 else pressure_now

    if idx >= 6:
        temp_6h_avg = df.loc[idx - 6:idx - 1, "temperature_2m"].mean()
        temp_6h_std = df.loc[idx - 6:idx - 1, "temperature_2m"].std()
        humidity_6h_avg = df.loc[idx - 6:idx - 1, "relative_humidity_2m"].mean()
        pressure_6h_avg = df.loc[idx - 6:idx - 1, "surface_pressure"].mean()
        rain_6h_sum = df.loc[idx - 6:idx - 1, "rain"].sum()
    else:
        temp_6h_avg = temp_now
        temp_6h_std = 0
        humidity_6h_avg = humidity_now
        pressure_6h_avg = pressure_now
        rain_6h_sum = 0

    if idx >= 24:
        rain_24h_sum = df.loc[idx - 24:idx - 1, "rain"].sum()
    else:
        rain_24h_sum = df.loc[0:idx - 1, "rain"].sum()

    rain_last_hour = 1 if (idx >= 1 and df.loc[idx - 1, "rain"] > 0) else 0
    rain_last_3h = 1 if (idx >= 3 and df.loc[idx - 3:idx - 1, "rain"].sum() > 0) else 0
    rain_last_6h = 1 if (idx >= 6 and df.loc[idx - 6:idx - 1, "rain"].sum() > 0) else 0

    pressure_change_1h = pressure_now - pressure_1h
    pressure_change_3h = pressure_now - pressure_3h
    pressure_change_6h = pressure_now - pressure_6h

    features = {
        "temperature_2m": temp_now,
        "relative_humidity_2m": humidity_now,
        "surface_pressure": pressure_now,
        "wind_speed_10m": current_weather["wind_speed"],
        "wind_dir_sin": np.sin(np.radians(current_weather["wind_deg"])),
        "wind_dir_cos": np.cos(np.radians(current_weather["wind_deg"])),
        "cloud_cover": current_weather["clouds"],

        "hour_sin": np.sin(2 * np.pi * now.hour / 24),
        "hour_cos": np.cos(2 * np.pi * now.hour / 24),
        "day_sin": np.sin(2 * np.pi * now.timetuple().tm_yday / 365),
        "day_cos": np.cos(2 * np.pi * now.timetuple().tm_yday / 365),
        "month": now.month,
        "is_winter": int(now.month in [12, 1, 2]),
        "is_summer": int(now.month in [6, 7, 8]),

        "temp_1h_ago": temp_1h,
        "temp_3h_ago": temp_3h,
        "temp_6h_ago": temp_6h,
        "temp_change_1h": temp_now - temp_1h,
        "temp_change_3h": temp_now - temp_3h,
        "temp_change_6h": temp_now - temp_6h,

        "humidity_1h_ago": humidity_1h,
        "humidity_change_1h": humidity_now - humidity_1h,

        "pressure_1h_ago": pressure_1h,
        "pressure_3h_ago": pressure_3h,
        "pressure_6h_ago": pressure_6h,
        "pressure_change_1h": pressure_change_1h,
        "pressure_change_3h": pressure_change_3h,
        "pressure_change_6h": pressure_change_6h,

        "temp_6h_avg": temp_6h_avg,
        "temp_6h_std": temp_6h_std if not np.isnan(temp_6h_std) else 0,
        "humidity_6h_avg": humidity_6h_avg,
        "pressure_6h_avg": pressure_6h_avg,

        "rain_6h_sum": rain_6h_sum,
        "rain_24h_sum": rain_24h_sum,
        "rain_last_hour": rain_last_hour,
        "rain_last_3h": rain_last_3h,
        "rain_last_6h": rain_last_6h,

        "pressure_falling": int(pressure_change_3h < -1),
        "pressure_rising": int(pressure_change_3h > 1),
    }

    current_df = pd.DataFrame([features])[FEATURE_COLUMNS]

    rain_probability = model.predict_proba(current_df)[0][1]
    rain_prediction = model.predict(current_df)[0]

    return rain_probability, rain_prediction


# 9. HELPER: WIND DIRECTION
def get_wind_direction_text(degrees):
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]
    index = round(degrees / 22.5) % 16
    return directions[index]


# 10. GENERATE WEATHER-BASED RECOMMENDATIONS
def generate_recommendations(current_weather, forecast_data, rain_probability):
    recommendations = []

    temp = current_weather["current_temp"]
    humidity = current_weather["humidity"]
    pressure = current_weather["pressure"]

    if rain_probability > 0.7:
        recommendations.append("High chance of rain. Don't forget your umbrella!")
    elif rain_probability > 0.5:
        recommendations.append("Moderate chance of rain. You might want to take an umbrella just in case.")
    elif rain_probability > 0.3:
        recommendations.append("Low chance of rain, but it's still a good idea to check the forecast before heading out.")

    if forecast_data and forecast_data["current_idx"] >= 3:
        df = forecast_data["df"]
        idx = forecast_data["current_idx"]
        pressure_change_3h = df.loc[idx - 3, 'surface_pressure']
        pressure_drop = pressure_change_3h - pressure

        if pressure_drop > 5:
            recommendations.append("Significant pressure drop detected. A storm might be approaching. Stay safe!")

    if temp > 35:
        recommendations.append("It's very hot outside. Stay hydrated and avoid prolonged sun exposure.")
    elif temp > 30:
        recommendations.append("It's quite warm. Make sure to drink water and take breaks if you're outside.")
    elif temp < 5:
        recommendations.append("It's cold outside. Dress warmly and consider wearing layers.")
    elif temp < 10:
        recommendations.append("It's a bit chilly. A light jacket might be a good idea.")

    if humidity > 80:
        recommendations.append("High humidity can make it feel hotter than it is. Stay cool and hydrated!")
    elif humidity < 30:
        recommendations.append("Low humidity can cause dry skin and discomfort. Consider using a moisturizer.")

    if current_weather['wind_speed'] > 40:
        recommendations.append("It's very windy outside. Be cautious if you're near trees or loose objects.")
        recommendations.append("Consider rescheduling outdoor activities if possible.")

    if not recommendations:
        recommendations.append("Weather looks good! Enjoy your day!")

    return recommendations


# 11. SHARED RENDERING LOGIC
def _render_weather(request, current_weather, city_label):
    lat = current_weather["lat"]
    lon = current_weather["lon"]

    forecast_data = fetch_forecast_and_history(lat, lon)

    rain_model = get_or_train_model(lat, lon)

    num_forecast_hours = 8

    if forecast_data:
        df = forecast_data['df']
        idx = forecast_data['current_idx']

        future_idx = range(idx + 1, min(idx + 1 + num_forecast_hours, len(df)))
        future_times = [df.loc[i, 'time'].strftime('%H:%M') for i in future_idx]
        future_temps = [round(df.loc[i, 'temperature_2m'], 1) for i in future_idx]
        future_humidity = [int(df.loc[i, 'relative_humidity_2m']) for i in future_idx]

        while len(future_times) < num_forecast_hours:
            future_times.append("--:--")
            future_temps.append(current_weather['current_temp'])
            future_humidity.append(current_weather['humidity'])
    else:
        tz_offset = current_weather.get('timezone_offset', 0)
        now = datetime.now(timezone.utc) + timedelta(seconds=tz_offset)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:%M") for i in range(num_forecast_hours)]
        future_temps = [current_weather['current_temp']] * num_forecast_hours
        future_humidity = [current_weather['humidity']] * num_forecast_hours

    if rain_model is not None:
        rain_prob, rain_pred = predict_rain_probability(
            rain_model, current_weather, forecast_data
        )
    else:
        if forecast_data:
            df = forecast_data['df']
            idx = forecast_data['current_idx']
            next_6h = range(idx, min(idx + 6, len(df)))
            rain_prob = max(df.loc[i, 'precipitation_probability'] for i in next_6h) / 100
        else:
            rain_prob = 0.3
        rain_pred = 1 if rain_prob > 0.5 else 0

    recommendations = generate_recommendations(current_weather, forecast_data, rain_prob)

    trend_data = get_trend_data(forecast_data)

    tz_offset = current_weather.get('timezone_offset', 0)
    local_time = datetime.now(timezone.utc) + timedelta(seconds=tz_offset)

    context = {
        'location': city_label,
        'city': current_weather['city'],
        'country': current_weather['country'],

        'current_temp': current_weather['current_temp'],
        'feels_like': current_weather['feels_like'],
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'humidity': current_weather['humidity'],
        'clouds': current_weather['clouds'],
        'description': current_weather['description'],
        'main_weather': current_weather['main_weather'],
        'icon': current_weather['icon'],

        'wind_speed': current_weather['wind_speed'],
        'wind_deg': current_weather['wind_deg'],
        'wind_direction': get_wind_direction_text(current_weather['wind_deg']),

        'pressure': current_weather['pressure'],
        'visibility': round(current_weather['visibility'] / 1000, 1),

        'sunrise': current_weather['sunrise'],
        'sunset': current_weather['sunset'],
        'day_length': current_weather['day_length'],

        'date': local_time.strftime("%B %d, %Y"),

        'rain_probability': round(rain_prob * 100),
        'rain_prediction': 'Yes' if rain_pred == 1 else 'No',

        'forecast_times': future_times,
        'forecast_temps': future_temps,
        'forecast_humidity': future_humidity,
        'forecast_hours': list(zip(future_times, future_temps, future_humidity)),

        'recommendations': recommendations,

        'lat': lat,
        'lon': lon,

        'model_type': 'HistGradientBoosting (Real Trends)',
        'data_source': 'Open-Meteo (365 days hourly)',

        'trend_times': trend_data['times'] if trend_data else [],
        'trend_temps': trend_data['temps'] if trend_data else [],
        'trend_humidities': trend_data['humidities'] if trend_data else [],
        'trend_current_pos': trend_data['current_position'] if trend_data else 0,
    }
    return render(request, 'weather.html', context)


# 12. MAIN VIEW FUNCTION
def weather_view(request):
    if request.method == 'POST':
        lat = request.POST.get('lat', '').strip()
        lon = request.POST.get('lon', '').strip()
        city = request.POST.get('city', '').strip()
        display_name = request.POST.get('display_name', '').strip()

        if lat and lon:
            try:
                lat_f, lon_f = float(lat), float(lon)
                if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
                    raise ValueError("Invalid coordinates")
                current_weather = get_current_weather_by_coords(lat_f, lon_f)
                label = display_name or current_weather['city']
                return _render_weather(request, current_weather, label)
            except (ValueError, TypeError):
                return render(request, 'weather.html', {'error': 'Invalid location coordinates'})
            except Exception as e:
                print(f"Coord weather error: {e}")
                return render(request, 'weather.html', {'error': str(e), 'location': city or display_name})

        if not city:
            return render(request, 'weather.html', {'error': 'Please enter a city name or use current location'})

        try:
            current_weather = get_current_weather(city)
            return _render_weather(request, current_weather, city)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return render(request, 'weather.html', {'error': str(e), 'location': city})

    lat = request.GET.get('lat')
    lon = request.GET.get('lon')
    if lat and lon:
        try:
            lat_f, lon_f = float(lat), float(lon)
            if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
                raise ValueError("Invalid coordinates")
            current_weather = get_current_weather_by_coords(lat_f, lon_f)
            geo = reverse_geocode(lat_f, lon_f)
            label = geo.get('display_name', current_weather['city'])
            return _render_weather(request, current_weather, label)
        except (ValueError, TypeError):
            return render(request, 'weather.html', {'error': 'Invalid location coordinates'})
        except Exception as e:
            print(f"Geolocation weather error: {e}")
            return render(request, 'weather.html', {'error': str(e)})

    return render(request, 'weather.html')
