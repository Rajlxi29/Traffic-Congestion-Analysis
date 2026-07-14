import numpy as  np
import pandas as pd

def time_change(df):
  dfcopy = df.copy()
  dfcopy["time_minutes"] = pd.to_datetime(dfcopy["timestamp"], format="%H:%M").dt.hour*60 + pd.to_datetime(dfcopy["timestamp"], format="%H:%M").dt.minute
  dfcopy["hour"] = dfcopy["time_minutes"]/60
  dfcopy["sin_hour"] = np.sin(2 * np.pi * dfcopy["hour"]/24)
  dfcopy["cos_hour"] = np.cos(2 * np.pi * dfcopy["hour"]/24)
  dfcopy["sin_minute"] = np.sin(2 * np.pi * dfcopy["time_minutes"]/1440)
  dfcopy["cos_minute"] = np.cos(2 * np.pi * dfcopy["time_minutes"]/1440)
  dfcopy["is_rush_hour"] = dfcopy["hour"].isin([8, 9, 10]).astype(int)
  dfcopy["night_time"] = dfcopy["hour"].isin([23, 1, 2, 4, ]).astype(int)

  return dfcopy

def Lag(dfn, df):
  dfncopy = dfn.copy()
  dfncopy = dfn.sort_values(["geohash", "day", "timestamp"])
  dfncopy["demand_lag_15"] = df.groupby("geohash")["demand"].shift(1)
  dfncopy["demand_lag_hour"] = df.groupby("geohash")["demand"].shift(4)
  dfncopy["demand_lag_day"] = df.groupby("geohash")["demand"].shift(96)
  dfncopy["demand_roll_mean_hour"] = df.groupby("geohash")["demand"].shift(1).rolling(4).mean()
  dfncopy["demand_roll_std_hour"] = df.groupby("geohash")["demand"].shift(1).rolling(4).std()