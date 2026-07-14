from django.shortcuts import render

import os
import requests
import pandas as pd
import numpy as np
import joblib

from datetime import datetime, timedelta, timezone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score

# Optional: File Locking for thread safety during model caching. 
try:
    from filelock import FileLock
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False
    print("Warning: filelock module not found. Install with pip install fileLock")


OPENWEATHERMAP_API_KEY = os.getenv("OWM_API_KEY", "decb14ec0acc76cbccae785f023d00c9") # CONFIRM THIS 
OPENWEATHERMAP_BASE_URL = "https://api.openweathermap.org/data/2.5/" # CONFIRM THIS

# Model-cache folder. This is where we will save the trained model and load it later for predictions.
MODEL_DIR = os.path.join(os.getcwd(), "models") # The trained model will be saved inside this folder.
os.makedirs(MODEL_DIR, exist_ok=True) # If the models folder does not exist, create it.

# ---------------------------------------------
# 1. FETCH CURRENT WEATHER FROM OPENWEATHERMAP
# ---------------------------------------------
def get_current_weather(city):
    url = f"{OPENWEATHERMAP_BASE_URL}weather?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"

    response = requests.get(url, timeout=10)

    data = response.json()

     # Handle wrong city name or API error
    if response.status_code != 200:
        raise Exception(data.get("message", "City not found"))
    
    # Get timezone offset from API
    timezone_offset = data.get("timezone", 0)

    # OpenWeatherAPI return time in UTC so Convert sunrise and sunset times from UTC to local time using the timezone offset
    sunrise_utc = datetime.fromtimestamp(data["sys"]["sunrise"], tz=timezone.utc)
    sunset_utc = datetime.fromtimestamp(data["sys"]["sunset"], tz=timezone.utc)

    local_tz = timezone(timedelta(seconds=timezone_offset))
    sunrise_local = sunrise_utc.astimezone(local_tz)
    sunset_local = sunset_utc.astimezone(local_tz)

    # Calculate day length
    day_length = (sunset_local - sunrise_local).seconds / 3600

    # Return cleaned weather information
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
        "wind_speed": round(data["wind"]["speed"] * 3.6, 1), # OpenWeatherMap gives wind speed in m/s. multiply by 3.6 to convert it to km/h.
        "wind_deg": data["wind"].get("deg", 0),
        "lat": data["coord"]["lat"], # Latitude and longitude are needed for Open-Meteo.
        "lon": data["coord"]["lon"],
        "sunrise": sunrise_local.strftime("%H:%M"),
        "sunset": sunset_local.strftime("%H:%M"),
        "day_length": round(day_length, 1),
        "timezone_offset": timezone_offset,
    }

# ------------------------------------------------------
# 2. FETCH RECENT WEATHER AND FORECAST FROM  OPEN METEO
# ------------------------------------------------------
# This function gets:
# - the past 24 hours of weather
# - the next 2 days forecast
def fetch_forecast_and_history(lat, lon, hours_forcast=12):
    url = "https://api.open-meteo.com/v1/forecast" # CONFIRM THE URL

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
            "cloudcover",
            "windspeed_10m",
            "winddirection_10m",
        ],
        "past_days": 1,  # CRITICAL: past_days=1 gives us recent real weather history.
        "forecast_days": 2, # forecast_days=2 gives us future hourly forecast.
        "timezone": "auto", # timezone=auto makes time match the city location.
    }
    try: 
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        # Convert hourly data into a pandas table
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

        # Find the current hour inside the dataframe
        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        # Convert time to timezone-naive so comparison will work.
        df["time_naive"] = df["time"].dt.tz_localize(None)
        current_idx = (df["time_naive"] - current_hour).abs().idxmin()

        return {
            "df": df,
            "current_idx": current_idx,
            "timezone": data.get("timezone", "UTC"),
        }
    except Exception as e:
        print("Error fetching forcast and history: {e} ")
        return None

# --------------------------------------------------------------------------------
# 3. FETCH HOURLY HISTORICAL WEATHER DATA FROM OPEN-METEO (FOR TRAINING THE MODEL)
# --------------------------------------------------------------------------------
# Reason for 1 full year is for seasonality. Weather patterns can be seasonal, 
# and having a full year of data allows the model to learn from all seasons, 
# improving its ability to generalize and make accurate predictions throughout the year.
def fetch_historical_data(lat, lon, days_back=365): #Requested 365 historical days of Hourly data. 
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back)

    url = "https://archive-api.open-meteo.com/v1/archive" # CONFIRM THE URL

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

        # Resampled to ensure continuous hourly data. This will fill in any missing hours with NaN, which we will drop later.
        df = df.asfreq('h') # Resampling to hourly frequency to ensure we have a continuous time series. This will introduce NaN for any missing hours, which we will handle next.
        df = df.ffill(limit=6) # Forward fill up to 6 hours of missing data. This is to handle occasional API gaps without losing too much data.
        df = df.dropna() # Drop any remaining rows with NaN values. After forward filling, there should be very few (if any) NaNs left, and dropping them will ensure our model gets clean data.
        df = df.reset_index() # Reset index to make 'time' a column again.

        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None
    
# -----------------------
# 4. FEAUTURE ENGINEERING
# -----------------------
def engineer_features(df):
    """
    Create ML features from hourly data.
    All features use ONLY past data (no future leakage) and are designed to help the model learn patterns that lead to rain in the next 6 hours.
    """

    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    # Target: Rain in next 6 hours.
    df["rain_next_6h"] = (
        df["rain"].rolling(window=6, min_periods=1).sum().shift(-6) > 0.1
    ).astype(int)

    # Time-based features
    # Also Covert them into cyclic features using sine and cosine transformations soo the model can underrstand Daily and seasonal cycles.
    df["hour"] = df["time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_year"] = df["time"].dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["month"] = df["time"].dt.month
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

    # Lag features for temperature, humidity, and pressure
    # makes the model understand past values which makes the model learn if patterns changes over time.
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

    # Rolling statistics to capture recent trends
    # This helps the model understand recent trends and variability, which can be important for predicting rain. 
    # For example, a sudden drop in pressure or a spike in humidity over the last few hours might indicate an approaching storm.
    df["temp_6h_avg"] = df["temperature_2m"].rolling(6).mean().shift(1)
    df["temp_6h_std"] = df["temperature_2m"].rolling(6).std().shift(1)
    df["humidity_6h_avg"] = df["relative_humidity_2m"].rolling(6).mean().shift(1)
    df["pressure_6h_avg"] = df["surface_pressure"].rolling(6).mean().shift(1)
    df["rain_6h_sum"] = df["rain"].rolling(6).sum().shift(1)
    df["rain_24h_sum"] = df["rain"].rolling(24).sum().shift(1)

    # Recent Rain Indicators
    # To indicate wether it rained recently and wether pressure is failing or rising. 
    # This can help the model learn patterns like 
    # "if it rained in the last hour, it's more likely to rain in the next 6 hours" or "if pressure is falling, rain is more likely".
    df["rain_last_hour"] = (df["rain"].shift(1) > 0).astype(int)
    df["rain_last_3h"] = (df["rain"].rolling(3).sum().shift(1) > 0).astype(int)
    df["rain_last_6h"] = (df["rain"].rolling(6).sum().shift(1) > 0).astype(int)

    # Wind features
    df["wind_dir_sin"] = np.sin(np.radians(df["wind_direction_10m"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["wind_direction_10m"]))

    # Pressure trend features (key predictor!)
    df["pressure_falling"] = (df["pressure_change_3h"] < -1).astype(int)
    df["pressure_rising"] = (df["pressure_change_3h"] > 1).astype(int)

    # Drop NaN values to drop some rows to have a clean data for training.
    df = df.dropna()

    return df

# ------------------------
# 5. Prepare Training Data
# ------------------------  
FEATURE_COLUMNS = [
    #The Inputs the model will use to learn patterns.

    #Current conditions
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_dir_sin", "wind_dir_cos", "cloud_cover",

    # Time features
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "month", "is_winter", "is_summer",

    # REAL historical features (the key to accurate predicition)
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
    """Prepare features (X) and target (y) for model training."""
    X = df[FEATURE_COLUMNS]
    y = df["rain_next_6h"]
    return X, y


# ----------------------------------
# 6. Train the Rain Prediction Model
# ----------------------------------
#Functions:
#- Uses HistGradientBoostingClassifier (faster, handles missing data)
#- Uses F1 score (handles class imbalance better than accuracy)
#- Uses class_weight to handle imbalanced data
def train_rain_model(X, y):
    # ------------------------------------------------------------
    # FALLBACK CHECK
    # ------------------------------------------------------------
    # If y has only one class, the real ML model cannot train.
    # Example: all values are 0 or all values are 1.
    # So we use DummyModel to prevent the app from crashing.
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
    
    # Calculate class weights to handle imbalance
    # This is important because rain events are often much less frequent than non-rain events, and without class weighting, 
    # the model might just learn to always predict "no rain" to achieve high accuracy, but it would fail to predict actual rain events. 
    # By using class_weight="balanced", we tell the model to give more importance to the minority class (rain) during training, 
    # which helps it learn to recognize patterns that lead to rain more effectively.
    n_smaples = len(y) # Total number of samples in the training data.
    n_positive = y.sum() # Number of samples where rain is expected in the next 6 hours (the positive class).
    n_negative = n_smaples - n_positive # Number of samples where rain is not expected (the negative class).

    # HistGradientBoostingClassifier (sklearn built-in, fast, handles NaN).
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
        class_weight="balanced" # This automatically adjusts weights inversely proportional to class frequencies in the input data, 
        #which helps the model learn from imbalanced datasets where rain events are less frequent than non-rain events.
    )

    # Time series cross-validation to evaluate model performance on unseen data. 
    # This is important to ensure that our model is not overfitting and can generalize well to new data.
    tscv = TimeSeriesSplit(n_splits=5)

    f1_scores = [] # This list will store the F1 scores for each fold of the time series cross-validation. F1 score is a good metric for imbalanced classification problems like ours, where rain events are less frequent than non-rain events. It considers both precision and recall, giving us a better sense of how well our model is predicting rain events without being biased by the majority class.
    acc_scores = [] # This list will store the accuracy scores for each fold of the time series cross-validation. Accuracy can be misleading in imbalanced datasets, but it's still useful to track alongside F1 score to get a complete picture of model performance.

    for train_idx, test_idx in tscv.split(X): # This loop performs time series cross-validation. It splits the data into 5 sequential folds, where each fold is used as a test set while the previous folds are used for training. This allows us to evaluate how well our model generalizes to unseen data over time, which is crucial for time series data like weather.
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] # This splits the feature data (X) into training and testing sets based on the indices provided by TimeSeriesSplit. The training set consists of all data up to a certain point in time, and the test set consists of the subsequent data, ensuring that we are always predicting future data from past data.
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx] # This splits the target variable (y) into training and testing sets corresponding to the feature splits. The model will be trained on y_train and evaluated on y_test to see how well it predicts rain in the next 6 hours based on the features in X_test.

        if y_train.nunique() < 2:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        acc_scores.append(accuracy_score(y_test, y_pred))

    # Final training on the entire dataset after cross-validation. This allows the model to learn from all available data before we save it for future predictions.
    model.fit(X, y)

    print("\nRain Model Performance")
    # Performance matrics averaged across all cross-validation folds. 
    # This makes the model more realistic beacause it learns patterns overtime instead of random data split that generalize over time instead of just memorizing the training data.
    metrics = {
        'f1': np.mean(f1_scores) if f1_scores else 0.5,
        'accuracy': np.mean(acc_scores) if acc_scores else 0.5
    }
    print(f"Rain Model - F1: {metrics['f1']:.2%}, Accuracy: {metrics['accuracy']:.2%}")

    return model, metrics


# ----------------------------------
# 7. MODEL CACHING WITH FILE LOCKING
# ----------------------------------
def get_or_train_model(lat, lon):
    # Load cached model or train a new one with file locking if cache is old or missing.
    cache_key = f"{round(lat, 1)}_{round(lon, 1)}" # To create a unique cache key based on each location(lon,lat), rounded to 1 decimal place. This allows us to have a separate model for each location while still keeping the cache manageable (rounding to 1 decimal means we are grouping locations within about 11 km together, which is reasonable for weather patterns).
    model_path = os.path.join(MODEL_DIR, f"rain_model_{cache_key}.joblib")
    lock_path = os.path.join(MODEL_DIR, f"rain_model_{cache_key}.lock")

    def load_model():
        if os.path.exists(model_path):
            # Check if the model file exist or is less than 24 hours old. If it is, we can load it instead of retraining.
            age = datetime.now().timestamp() - os.path.getmtime(model_path)

            if age < 86400: # If the model is less than 24 hours old, load it.
                print("\nLoading saved model...")
                return joblib.load(model_path)
            return None

    def train_and_save_model(): 
        print("\nFetching 365 days historical weather data...")
        df = fetch_historical_data(lat, lon, days_back=365)

        if df is None or len(df) < 100: # If we don't have enough data, we cannot train a good model. So we return None and the app will use the fallback DummyModel.
            print("Not enough historical data to train model.")
            return None

        print(f"Engineering features from {len(df)} hourly records...")
        df = engineer_features(df)
        if len(df) < 100: # After feature engineering, we might lose some rows due to NaN values. If we have less than 100 rows left, it's not enough to train a good model.
            print("Not enough data after feature engineering to train model.")
            return None

        print("Training improved rain prediction model with HistGradientBoosting...") # This is the main training step where we create and evaluate the machine learning model using the historical weather data and the engineered features. The model learns to predict whether it will rain in the next 6 hours based on patterns in the past weather data.  
        X, y = prepare_training_data(df)
        model = train_rain_model(X, y)
        
        joblib.dump(model, model_path)
        print("Model saved successfully.")

        return model
    
    # Saved Model using Joblib for caching.
    model = load_model()
    if model is not None:
        return model
    
    # File Locking to prevent multiple processes from training the model at the same time. 
    # This is important in a web app where multiple users might trigger model training simultaneously. 
    # The lock file acts as a signal that a training process is already running, 
    # so other processes will wait until the lock is released before they can train or load the model.
    # This prevents duplicate work and potential file corruption from multiple processes trying to write to the same model file at the same time.
    # This what makes the system production ready optimizing both performance and reliability.
    if HAS_FILELOCK: # If we have filelock library, we can use it for better locking mechanism.
        with FileLock(lock_path, timeout=180): # Wait up to 3 minutes for the lock. This is to handle cases where training might take a long time, especially if the historical data is large.
            model = load_model() # After acquiring the lock, check again if the model was trained by another process while we were waiting for the lock. If it was, we can load it instead of retraining.
            if model is not None:
                return model
            return train_and_save_model() # If the model still does not exist, we proceed to train and save it.
    else: # If we don't have filelock library, we use a simple lock file mechanism.
        return train_and_save_model()
    
# -------------------------------------------------------
# 8. MAKING THE PREDICTION FUNCTION (WITH REAL-TIME DATA)
# -------------------------------------------------------
def predict_rain_probability(model, current_weather, forecast_data):
    if forecast_data is None:
        print("No forecast data available for prediction.")
        return 0.3, 0 # Return a default probability and prediction if we don't have forecast data. This is a fallback to ensure the app can still function even if the forecast data fetch fails.
    
    df = forecast_data["df"]
    idx = forecast_data["current_idx"]

    # Ensure we have enough past data to calculate features. If not, return a default probability and prediction. 
    # This is important because the feature engineering relies on past data, 
    # and if we don't have enough of it (e.g., if the API just started providing data), 
    # we cannot calculate the features properly. In this case, we return a default value to keep the app running without crashing.
    if idx < 6:
        return 0.3, 0

    now = datetime.now()

    #Extracct Real-time features from actual values (No Guessing) to make the prediction more accurate and up to date.
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

    # Caluculate rolling statistics for the past 6 hours to capture recent trends, which can be important for predicting rain.
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

    # Recent Rain from REAL Data
    rain_last_hour = 1 if (idx >= 1 and df.loc[idx - 1, "rain"] > 0) else 0
    rain_last_3h = 1 if (idx >= 3 and df.loc[idx - 3:idx - 1, "rain"].sum() > 0) else 0
    rain_last_6h = 1 if (idx >= 6 and df.loc[idx - 6:idx - 1, "rain"].sum() > 0) else 0

    # Calculate REAL trends
    pressure_change_1h = pressure_now - pressure_1h
    pressure_change_3h = pressure_now - pressure_3h
    pressure_change_6h = pressure_now - pressure_6h
    
    # To build the feature vectors that matches the features we used for training the model. 
    # This is critical because the model expects the input features to be in the same format and order as the data it was trained on.
    # By using real-time data for these features, we ensure that our predictions are based on the most current weather conditions, 
    # which can significantly improve the accuracy of rain predictions.
    features = {
        "temperature_2m": temp_now,
        "relative_humidity_2m": humidity_now,
        "surface_pressure": pressure_now,
        "wind_speed_10m": current_weather["wind_speed"],
        "wind_dir_sin": np.sin(np.radians(current_weather["wind_deg"])),
        "wind_dir_cos": np.cos(np.radians(current_weather["wind_deg"])),
        "cloud_cover": current_weather["clouds"],

        # Time features (cyclic)
        "hour_sin": np.sin(2 * np.pi * now.hour / 24),
        "hour_cos": np.cos(2 * np.pi * now.hour / 24),
        "day_sin": np.sin(2 * np.pi * now.timetuple().tm_yday / 365),
        "day_cos": np.cos(2 * np.pi * now.timetuple().tm_yday / 365),
        "month": now.month,
        "is_winter": int(now.month in [12, 1, 2]),
        "is_summer": int(now.month in [6, 7, 8]),

        # REAL historical features (the key to accurate predicition) - Not based on guessing.
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

    # Create a DataFrame with the same feature columns as the training data. 
    # This ensures that the model receives the input in the correct format and can make accurate predictions based on the current weather conditions and recent historical data.
    current_df = pd.DataFrame([features])[FEATURE_COLUMNS] #!!

    # Pass through the model to get the probability of rain and the binary prediction (rain or no rain).
    rain_probability = model.predict_proba(current_df)[0][1] #!
    rain_prediction = model.predict(current_df)[0] #!

    return rain_probability, rain_prediction


#-------------------
# 9. Helper function
# ------------------ 
# to convert wind direction in degrees to compass direction (e.g., N, NE, E, etc.) for better readability in the UI.
def get_wind_direction_text(degrees):
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]

    index = round(degrees / 22.5) % 16 #!!
    return directions[index]

# ------------------------------------------
# 10. Generate Weather-based Recommendations
# ------------------------------------------ 
# based on the predicted rain probability and current weather conditions.
def generate_recommendations(current_weather, forecast_data, rain_probability):
    recommendations = []

    temp = current_weather["current_temp"]
    humidity = current_weather["humidity"]
    pressure = current_weather["pressure"]

    # Rain-based recommendations
    if rain_probability > 0.7:
        recommendations.append("High chance of rain. Don't forget your umbrella! ☔")
    elif rain_probability > 0.5:
        recommendations.append("Moderate chance of rain. You might want to take an umbrella just in case. ☂️")
    elif rain_probability > 0.3:
        recommendations.append("Low chance of rain, but it's still a good idea to check the forecast before heading out. 🌦️")
    
    # Check pressure trends for storm warnings
    if forecast_data and forecast_data["current_idx"] >= 3:
        df = forecast_data["df"]
        idx = forecast_data["current_idx"]
        pressure_change_3h = df.loc[idx - 3, 'surface_pressure']
        pressure_drop = pressure_change_3h - pressure

        if pressure_drop > 5:
            recommendations.append("Significant pressure drop detected. A storm might be approaching. Stay safe! 🌩️")
    
    # Temperature-based recommendations
    if temp > 35:
        recommendations.append("It's very hot outside. Stay hydrated and avoid prolonged sun exposure. 🥵")
    elif temp > 30:
        recommendations.append("It's quite warm. Make sure to drink water and take breaks if you're outside. 🌞")
    elif temp < 5:
        recommendations.append("It's cold outside. Dress warmly and consider wearing layers. 🧥")
    elif temp < 10:
        recommendations.append("It's a bit chilly. A light jacket might be a good idea. 🧣")
    
    # Humidity-based recommendations
    if humidity > 80:
        recommendations.append("High humidity can make it feel hotter than it is. Stay cool and hydrated! 💧")
    elif humidity < 30:
        recommendations.append("Low humidity can cause dry skin and discomfort. Consider using a moisturizer. 🌵")

    # Weather-based activity recommendations
    if current_weather['wind_speed'] > 40:
        recommendations.append("It's very windy outside. Be cautious if you're near trees or loose objects. 🌬️")
        recommendations.append("Consider rescheduling outdoor activities if possible. 🏞️")
    
    if not recommendations:
        recommendations.append("Weather looks good! Enjoy your day! 😊")

    return recommendations


def print_future_forecast(forecast_data):
    df = forecast_data["df"]
    idx = forecast_data["current_idx"]

    print("\nNext 5 Hours Forecast")
    print("----------------------")

    for i in range(1, 6):
        row = df.loc[idx + i]

        time = row["time"].strftime("%H:%M")
        temp = row["temperature_2m"]
        humidity = row["relative_humidity_2m"]
        rain_prob = row["precipitation_probability"]

        print(f"{time} | Temp: {temp}°C | Humidity: {humidity}% | Rain Chance: {rain_prob}%")

# ---------------------------------------------------------
# 11. MAIN VIEW FUNCTION TO DISPLAY WEATHER AND PREDICTIONS
# ---------------------------------------------------------
# That connects it all.
def weather_view(request):
    #First fetch the current weather data
    if request.method == 'POST':
        city = request.POST.get('city', '').strip()

        if not city:
            return render(request, 'weather.html', {'error': 'Please Enter A City Name'}) # This is a Django template rendering function that needs to be imported from django.shortcuts.

        try:
            # 1. Get Current Weather
            current_weather = get_current_weather(city)

            lat = current_weather["lat"]
            lon = current_weather["lon"]

            # 2. Get forecast + recent history (Critical for accurate prediction)
            forecast_data = fetch_forecast_and_history(lat, lon)

            # 3. Get or train ML model
            rain_model = get_or_train_model(lat, lon)

            # 4. Process the forecast data for the next few hours. 
            if forecast_data:
                df = forecast_data['df']
                idx = forecast_data['current_idx']

                # Get next 5 hours for display
                future_idx = range(idx +1, min(idx+6, len(df)))
                future_times = [df.loc[i, 'time'].strftime('%H:%M') for i in future_idx]
                future_temps = [round(df.loc[i, 'temperature_2m'], 1) for i in future_idx]
                future_humidity = [int(df.loc[i, 'relative_humidity_2m']) for i in future_idx]

                # Pad if needed
                while len(future_times) < 5:
                    future_times.append(" --:-- ")
                    future_temps.append(current_weather['current_temp'])
                    future_humidity.append(current_weather['humidity'])

            else:
                # Fallback 
                tz_offset = current_weather.get('timezone_offset', 0)
                now = datetime.now(timezone.utc) + timedelta(seconds=tz_offset)
                next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                future_times = [(next_hour + timedelta(hours=i)).strftime("%H:%M") for i in range(5)]
                future_temps = [current_weather['current_temp']] * 5
                future_humidity = [current_weather['humidity']] * 5

            # 5. Make rain prediction with Real data
            if rain_model is not None:
                rain_prob, rain_pred = predict_rain_probability(
                    rain_model, current_weather, forecast_data
                )
            else:
                # Use Open-Meteo's precipitation probability as fallback
                if forecast_data:
                    df = forecast_data['df']
                    idx = forecast_data['current_idx']
                    next_6h = range(idx, min(idx+ 6, len(df)))
                    rain_prob = max(df.loc[i, 'precipitation_probability'] for i in next_6h) / 100
                else:
                    rain_prob = 0.3

                rain_pred = 1 if rain_prob > 0.5 else 0
            #6. Get recomendations 
            recomendations = generate_recommendations(current_weather, forecast_data,  rain_prob)

            #7. Build Context
            tz_offset = current_weather.get('timezone_offset',0)
            local_time = datetime.now(timezone.utc) + timedelta(seconds=tz_offset)

            context = {
                # Location
                'location': city,
                'city': current_weather['city'],
                'country': current_weather['country'],

                # Current conditions
                'current_temp': current_weather['current_temp'],
                'feels_like': current_weather['feels_like'],
                'MinTemp': current_weather['temp_min'],
                'MaxTemp': current_weather['temp_max'],
                'humidity': current_weather['humidity'],
                'clouds': current_weather['clouds'],
                'description': current_weather['description'],
                'main_weather': current_weather['main_weather'],
                'icon': current_weather['icon'],

                # Wind
                'wind_speed': current_weather['wind_speed'],
                'wind_deg': current_weather['wind_deg'],
                'wind_direction': get_wind_direction_text(current_weather['wind_deg']),

                # Atmosphere
                'pressure': current_weather['pressure'],
                'visibility': round(current_weather['visibility'] / 1000,1),

                # Sun
                'sunrise': current_weather['sunrise'],
                'sunset': current_weather['sunset'],
                'day_length': current_weather['day_length'],

                # Time
                'date': local_time.strftime("%B %d, %Y"),

                # ML Predictions
                'rain_probability': round(rain_prob * 100),
                'rain_prediction': 'Yes' if rain_pred == 1 else 'No',

                # Hourly forecast
                'time1': future_times[0],
                'time2': future_times[1],
                'time3': future_times[2],
                'time4': future_times[3],
                'time5': future_times[4],

                'temp1': future_temps[0],
                'temp2': future_temps[1],
                'temp3': future_temps[2],
                'temp4': future_temps[3],
                'temp5': future_temps[4],

                'hum1': future_humidity[0],
                'hum2': future_humidity[1],
                'hum3': future_humidity[2],
                'hum4': future_humidity[3],
                'hum5': future_humidity[4],

                # Recommendations
                'recommendations': recomendations,

                # Model info
                'model_type': 'HistGradientBoosting (Real Trends)',
                'data_source': 'Open-Meteo (365 days hourly)',
            }
            return render(request, 'weather.html', context) #!!
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return render(request, 'weather.html', {'error': str(e), 'location': city}) #!!
        
    return render(request, 'weather.html') #!!