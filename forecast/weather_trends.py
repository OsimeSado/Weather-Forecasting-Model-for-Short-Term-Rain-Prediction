def get_trend_data(forecast_data, hours_back=24):
    if forecast_data is None:
        return None

    df = forecast_data["df"]
    idx = forecast_data["current_idx"]

    start = max(0, idx - hours_back)
    end = min(idx + 1, len(df))

    if end - start < 3:
        return None

    subset = df.iloc[start:end].copy()

    times = subset["time"].dt.strftime("%H:%M").tolist()
    temps = [round(float(v), 1) for v in subset["temperature_2m"]]
    humidities = [int(v) for v in subset["relative_humidity_2m"]]

    current_position = len(times) - 1

    return {
        "times": times,
        "temps": temps,
        "humidities": humidities,
        "current_position": current_position,
    }
