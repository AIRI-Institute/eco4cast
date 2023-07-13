import datetime
import pandas as pd
import requests
import json

parameters = [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "surface_pressure",
    "cloudcover",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    "windspeed_10m",
    "winddirection_10m",
    "windgusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    # "direct_normal_irradiance",
    "diffuse_radiation",
    "vapor_pressure_deficit",
    "et0_fao_evapotranspiration",
    "precipitation",
    "snowfall",
    "rain",
    "weathercode",
]



def get_last_weather_data(latitude, longitude, lookback_days=14, parameters=parameters):
    response_string = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&past_days={lookback_days}&forecast_days=1"
    response_string += f"&hourly="
    for i, parameter in enumerate(parameters):
        if i != len(parameters):
            response_string += parameter + ","
        else:
            response_string += parameter
    response = requests.get(response_string).content
    weather_dictionary = json.loads(response)
    dataframe = pd.DataFrame.from_dict(weather_dictionary["hourly"])
    dataframe["time"] = pd.to_datetime(dataframe["time"])
    current_time = datetime.datetime.now(datetime.timezone.utc)
    dataframe = dataframe[dataframe["time"].dt.tz_localize("UTC") < current_time]

    return dataframe


