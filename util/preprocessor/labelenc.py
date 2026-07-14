import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def labelEncode(df):
  df_copy = df.copy()

  df_copy["Temprature"] = df_copy["Temprature"].fillna(np.mean(df_copy["Temprature"]))
  df_copy["NumberofLanes"] = df_copy["NumberofLanes"].fillna(df_copy["NumberofLanes"].median())

  df_copy = df_copy.drop("timestamp", axis = 1)
  return df_copy

def preprocess(df, le_geohash=None, le_large_vehicles=None, le_landmarks=None, oh_roadtype=None, oh_weather=None, fit_encoders=False):
    df_copy = df.copy()

    if fit_encoders:
        le_geohash = LabelEncoder()
        le_large_vehicles = LabelEncoder()
        le_landmarks = LabelEncoder()
        oh_roadtype = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        oh_weather = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        df_copy["LargeVehicles"] = le_large_vehicles.fit_transform(df_copy["LargeVehicles"])
        df_copy["Landmarks"] = le_landmarks.fit_transform(df_copy["Landmarks"])
        df_copy["geohash"] = le_geohash.fit_transform(df_copy["geohash"])

        encoded = oh_roadtype.fit_transform(df_copy[["RoadType"]])
        encoded_w = oh_weather.fit_transform(df_copy[["Weather"]])
    else:
        if not all([le_geohash, le_large_vehicles, le_landmarks, oh_roadtype, oh_weather]):
            raise ValueError("Encoders must be provided when fit_encoders is False.")

        df_copy["LargeVehicles"] = le_large_vehicles.transform(df_copy["LargeVehicles"])
        df_copy["Landmarks"] = le_landmarks.transform(df_copy["Landmarks"])
        df_copy["geohash"] = le_geohash.transform(df_copy["geohash"])

        encoded = oh_roadtype.transform(df_copy[["RoadType"]])
        encoded_w = oh_weather.transform(df_copy[["Weather"]])

    encoded_arr = pd.DataFrame(encoded, columns=oh_roadtype.get_feature_names_out(["RoadType"]), index=df_copy.index)
    encoded_w_arr = pd.DataFrame(encoded_w, columns=oh_weather.get_feature_names_out(["Weather"]), index=df_copy.index)

    df_copy = df_copy.drop(["Weather", "RoadType"], axis=1)
    df_copy = pd.concat([df_copy, encoded_arr, encoded_w_arr], axis=1)

    return df_copy, le_geohash, le_large_vehicles, le_landmarks, oh_roadtype, oh_weather

def preprocessT(df, le_geohash=None, le_large_vehicles=None, le_landmarks=None, oh_roadtype=None, oh_weather=None):
  df_copy = df.copy()
  if not all([le_geohash, le_large_vehicles, le_landmarks, oh_roadtype, oh_weather]):
    raise ValueError("Encoders must be provided when fit_encoders is False.")

  df_copy["LargeVehicles"] = le_large_vehicles.transform(df_copy["LargeVehicles"])
  df_copy["Landmarks"] = le_landmarks.transform(df_copy["Landmarks"])
  df_copy["geohash"] = le_geohash.transform(df_copy["geohash"])

  encoded = oh_roadtype.transform(df_copy[["RoadType"]])
  encoded_w = oh_weather.transform(df_copy[["Weather"]])

  encoded_arr = pd.DataFrame(encoded, columns=oh_roadtype.get_feature_names_out(["RoadType"]), index=df_copy.index)
  encoded_w_arr = pd.DataFrame(encoded_w, columns=oh_weather.get_feature_names_out(["Weather"]), index=df_copy.index)

  df_copy = df_copy.drop(["Weather", "RoadType"], axis=1)
  df_copy = pd.concat([df_copy, encoded_arr, encoded_w_arr], axis=1)

  return df_copy, le_geohash, le_large_vehicles, le_landmarks, oh_roadtype, oh_weather