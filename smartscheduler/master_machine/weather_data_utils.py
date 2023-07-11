from sklearn.preprocessing import OneHotEncoder
import datetime
import numpy as np
import pandas as pd
import requests
import cartopy.io.shapereader as shpreader
from shapely import Point
from itertools import product
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


def get_historic_data(
    latitude: str,
    longitude: str,
    year_start: str,
    month_start: str,
    day_start: str,
    year_end: str,
    month_end: str,
    day_end: str,
    parameters: list,
):
    """
    This function gets historic weather data from https://open-meteo.com/en/docs/historical-weather-api#api_form
    Parameters - is a list of weather parameters you can need, for example, temperature, wind speed etc.
    """
    response_string = f"https://archive-api.open-meteo.com/v1/era5?latitude={latitude}&longitude={longitude}&start_date="
    response_string += f"{year_start}-{month_start}-{day_start}&end_date={year_end}-{month_end}-{day_end}&hourly="
    for i, parameter in enumerate(parameters):
        if i != len(parameters):
            response_string += parameter + ","
        else:
            response_string += parameter
    response = requests.get(response_string).content
    weather_dictionary = json.loads(response)
    dataframe = pd.DataFrame.from_dict(weather_dictionary["hourly"])
    dataframe["time"] = pd.to_datetime(dataframe["time"])
    return dataframe


def get_emission_data(
    year_start: str,
    month_start: str,
    day_start: str,
    year_end: str,
    month_end: str,
    day_end: str,
):
    """
    This function gets historic emission data from https://www.energidataservice.dk/tso-electricity/CO2Emis website
    """
    date = datetime.datetime(
        int(year_end), int(month_end), int(day_end)
    ) + datetime.timedelta(days=1)
    year_end, month_end, day_end = str(date.year), str(date.month), str(date.day)
    if len(month_end) != 2:
        month_end = "0" + month_end
    if len(day_end) != 2:
        day_end = "0" + day_end
    response_string = (
        f"https://api.energidataservice.dk/dataset/CO2Emis?offset=0&start="
    )
    response_string += f"{year_start}-{month_start}-{day_start}T00:00&end={year_end}-{month_end}-{day_end}T00:00&sort=Minutes5UTC%20DESC&timezone=dk"

    response = requests.get(response_string).content
    emission_dictionary = eval(response)
    # return emission_dictionary
    dataframe = pd.DataFrame.from_dict(emission_dictionary["records"])
    # update due to DK1 DK2 zones are now different
    dataframe = (
        dataframe.groupby(["Minutes5UTC", "Minutes5DK"])["CO2Emission"]
        .mean()[::-1]
        .reset_index()
    )
    # dataframe = dataframe[dataframe['PriceArea'] == 'DK1']
    # dataframe = dataframe.drop(columns=['PriceArea'])

    return dataframe


def average_emission_data(df: pd.DataFrame):
    """
    This function takes a dataframe with emission data with period of 5 minutes,
    process it, averaging by time, returning a new dataframe with emission data with period of 1 hour
    """
    df["Minutes5UTC"] = pd.to_datetime(df["Minutes5UTC"])
    df["Minutes5DK"] = pd.to_datetime(df["Minutes5UTC"])
    df = df.set_index("Minutes5DK")
    df = df.groupby(pd.Grouper(key="Minutes5UTC", freq="1H")).mean()
    # print(df.convert_dtypes().dtypes)
    return df.fillna(method="ffill")


def join_datasets(
    emission_dataframe: pd.DataFrame,
    weather_dataframe: pd.DataFrame,
    delta_days: int = 0,
):
    # delta = np.timedelta64(delta_days, 'D')
    #   print(delta)
    _, emission_indexes, weather_indexes = np.intersect1d(
        emission_dataframe.index.values,
        weather_dataframe["time"].values + np.timedelta64(delta_days, "D"),
        return_indices=True,
    )
    emission_dataframe = emission_dataframe.iloc[emission_indexes]
    weather_dataframe = weather_dataframe.iloc[weather_indexes]

    dataframe = pd.DataFrame(
        weather_dataframe.values, columns=weather_dataframe.columns
    )
    dataframe["emission"] = emission_dataframe["CO2Emission"].values
    return dataframe, emission_dataframe


def get_points_over_country(country_code: str, points_step: float):
    """
    This function makes a grid of points over a country.

    Args:
        country_code (str) : country ISO-3 code. Look for country code: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3
        points_step (float) : distance between points in a grid in degrees.

    Returns:
        List (longitude, latitude) : Evenly spaced points.

    """
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()

    for country in countries:
        if country.attributes["ADM0_A3"] == country_code:
            break

    min_longitude, min_latitude, max_longitude, max_latitude = country.geometry.bounds

    points = []
    for longitude, latitude in product(
        np.arange(min_longitude, max_longitude, points_step),
        np.arange(min_latitude, max_latitude, points_step),
    ):
        # checking points in the grid (within country boundaries) with step = points_step
        if Point((longitude, latitude)).within(country.geometry):
            points.append((longitude, latitude))
        else:
            # check if we can move point a bit
            max_shift = points_step / 5
            for long_dif, lat_dif in product(
                np.linspace(-max_shift, max_shift, 3), repeat=2
            ):
                if Point((longitude + long_dif, latitude + lat_dif)).within(
                    country.geometry
                ):
                    points.append((longitude + long_dif, latitude + lat_dif))
                    break

    return points


def get_multiple_historic_data(
    points,
    year_start,
    month_start,
    day_start,
    year_end,
    month_end,
    day_end,
    parameters=parameters,
    include_weathercode=False,
    co2_emission_delta_days=1,
):
    """
    Function gets historic data for each of the points from start time to end time. Timestep is 1 hour.

    if include_weathercode is False - drops "weathercode" feature due to it's categorical
    if include_weathercode is True - concats it as one-hot vector to other features

    if co2_emission_delta_days is None - doesnt include co2 info in matrix
    if co2_emission_delta_days is int - includes co2 info in features of each point with shift of co2_emission_delta_days

    Returns:
        np.array: [timesteps_num, features_num, points_num]

    Example 1. (Denmark with 40 points from 31.01.2021 to 02.03.2023, include_weathercode=False, co2_emission_delta_days=None) returns array with shape (18264, 21, 40)
    Example 2. (Denmark with 40 points from 31.01.2021 to 02.03.2023, include_weathercode=True, co2_emission_delta_days=1) returns array with shape (18239, 50, 40)
    """
    # info from https://open-meteo.com/en/docs
    openmeteo_weathercodes = [
        0,
        1,
        2,
        3,
        45,
        48,
        51,
        53,
        55,
        56,
        57,
        61,
        63,
        65,
        66,
        67,
        71,
        73,
        75,
        77,
        80,
        81,
        82,
        85,
        86,
        95,
        96,
        99,
    ]
    weather_matrices = []

    emission_df = get_emission_data(
        year_start=year_start,
        month_start=month_start,
        day_start=day_start,
        year_end=year_end,
        month_end=month_end,
        day_end=day_end,
    )
    emission_df = average_emission_data(emission_df)

    for longitude, latitude in points:
        weather_df = get_historic_data(
            latitude=latitude,
            longitude=longitude,
            year_start=year_start,
            month_start=month_start,
            day_start=day_start,
            year_end=year_end,
            month_end=month_end,
            day_end=day_end,
            parameters=parameters,
        )
        if "weathercode" in weather_df.columns:
            if not include_weathercode:
                # remove weathercode feature
                weather_df = weather_df.drop(columns=["weathercode"])
            else:
                # use weathercode feature as one-hot vector
                encoder = OneHotEncoder(
                    categories=[openmeteo_weathercodes],
                    sparse_output=False,
                )
                onehot_weathercodes = encoder.fit_transform(
                    weather_df["weathercode"].values.reshape(-1, 1)
                )
                weather_df = weather_df.drop(columns=["weathercode"])
                new_column_names = [
                    f"weathercode_{i}" for i in range(len(openmeteo_weathercodes))
                ]
                weather_df[new_column_names] = pd.DataFrame(
                    onehot_weathercodes, index=weather_df.index
                )

        if co2_emission_delta_days is not None:
            # use same co2 emission info for each point
            weather_df, emission_df = join_datasets(
                emission_df, weather_df, delta_days=co2_emission_delta_days
            )
        weather_matrix = weather_df.drop(columns=["time"]).to_numpy()
        weather_matrices.append(weather_matrix)

    return (
        np.array(weather_matrices).transpose((1, 2, 0)),
        weather_df["time"],
        emission_df,
    )


def get_last_weather_data(latitude, longitude, lookback_days=14, parameters=parameters):
    response_string = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&past_days={lookback_days}&forecast_days=1"
    response_string += f"&hourly="
    for i, parameter in enumerate(parameters):
        if i != len(parameters):
            response_string += parameter + ","
        else:
            response_string += parameter
    response = requests.get(response_string).content
    # print(response)
    weather_dictionary = eval(response)
    dataframe = pd.DataFrame.from_dict(weather_dictionary["hourly"])
    dataframe["time"] = pd.to_datetime(dataframe["time"])
    current_time = datetime.datetime.now(datetime.timezone.utc)
    dataframe = dataframe[dataframe["time"].dt.tz_localize("UTC") < current_time]

    return dataframe


def get_multiple_last_weather_data(
    points,
    lookback_window,
    parameters=parameters,
    include_weathercode=False,
    co2_emission_delta_days=1,
):
    """
    Function gets weather data for each of the points from current time to current time - lookback_window. Timestep is 1 hour.

    Args:
        points (List[longitude, latitude]) : list of points to get data
        lookback_window (int) : hours of lookback

        include_weathercode (bool) if False - drops "weathercode" feature due to it's categorical, else -
        concats it as one-hot vector to other features

        co2_emission_delta_days (bool) : if None - doesnt include co2 info in matrix, else -
        includes co2 info in features of each point with shift of co2_emission_delta_days

    Returns:
        np.array: [lookback_window, features_num, points_num]
    """
    # info from https://open-meteo.com/en/docs
    openmeteo_weathercodes = [
        0,
        1,
        2,
        3,
        45,
        48,
        51,
        53,
        55,
        56,
        57,
        61,
        63,
        65,
        66,
        67,
        71,
        73,
        75,
        77,
        80,
        81,
        82,
        85,
        86,
        95,
        96,
        99,
    ]
    weather_matrices = []

    current_time = datetime.datetime.now(datetime.timezone.utc)
    delta_days = 0 if co2_emission_delta_days is None else co2_emission_delta_days
    start_time = current_time - datetime.timedelta(
        days=delta_days, hours=lookback_window
    )

    # emission_df = None
    # if co2_emission_delta_days is not None:
    #     emission_df = get_emission_data(
    #         year_start=start_time.year,
    #         month_start=start_time.month
    #         if start_time.month > 9
    #         else f"0{start_time.month}",
    #         day_start=start_time.day if start_time.day > 9 else f"0{start_time.day}",
    #         year_end=current_time.year,
    #         month_end=current_time.month
    #         if current_time.month > 9
    #         else f"0{current_time.month}",
    #         day_end=current_time.day + 1
    #         if current_time.day > 8
    #         else f"0{current_time.day}",
    #     )
    #     emission_df = average_emission_data(emission_df)

    for longitude, latitude in points:
        weather_df = get_last_weather_data(
            latitude=latitude,
            longitude=longitude,
            lookback_days=(lookback_window // 24) + 2,
        )

        if "weathercode" in weather_df.columns:
            if not include_weathercode:
                # remove weathercode feature
                weather_df = weather_df.drop(columns=["weathercode"])
            else:
                # use weathercode feature as one-hot vector
                encoder = OneHotEncoder(
                    categories=[openmeteo_weathercodes],
                    sparse_output=False,
                )
                onehot_weathercodes = encoder.fit_transform(
                    weather_df["weathercode"].values.reshape(-1, 1)
                )
                weather_df = weather_df.drop(columns=["weathercode"])

                new_column_names = [
                    f"weathercode_{i}" for i in range(len(openmeteo_weathercodes))
                ]
                weather_df[new_column_names] = pd.DataFrame(
                    onehot_weathercodes, index=weather_df.index
                )

        # if co2_emission_delta_days is not None:
        #     # use same co2 emission info for each point
        #     weather_df, emission_df = join_datasets(
        #         emission_df, weather_df, delta_days=co2_emission_delta_days
        #     )
        weather_matrix = weather_df.drop(columns=["time"]).to_numpy()
        weather_matrices.append(weather_matrix)

    weather_data = np.array(weather_matrices).transpose((1, 2, 0))[-lookback_window:]
    weather_time_data = weather_df["time"][-lookback_window:]
    emission_df = None if emission_df is None else emission_df[-lookback_window:]

    return weather_data, weather_time_data, emission_df




if __name__ == "__main__":
    denmark_points = get_points_over_country("DNK", 0.5)
    assert len(denmark_points) == 40

    year_start = "2023"
    month_start = "01"
    day_start = "31"

    year_end = "2023"
    month_end = "03"
    day_end = "02"

    weather_points_dataset, datetimes, emission_df = get_multiple_historic_data(
        denmark_points,
        year_start=year_start,
        month_start=month_start,
        day_start=day_start,
        year_end=year_end,
        month_end=month_end,
        day_end=day_end,
        include_weathercode=True,
        co2_emission_delta_days=1,
    )

    assert datetimes.shape[0] == weather_points_dataset.shape[0]

    # if emission_df is None:
    #     emission_df = get_emission_data(
    #         year_start, month_start, day_start, year_end, month_end, day_end)
    #     emission_df = average_emission_data(emission_df)

    print(emission_df.shape, weather_points_dataset.shape)
    assert emission_df.shape[0] == weather_points_dataset.shape[0]
