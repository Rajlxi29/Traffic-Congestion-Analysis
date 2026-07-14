from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import TypedDict, Literal, List
from util.preprocessor.labelenc import labelEncode, preprocess, preprocessT
from util.preprocessor.temporal import time_change
from util.process_geohash import GeohashProcessor
import requests
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

df = pd.read_csv("dataset/LagShift.csv")

def getlag(geohash):
    print(df[df["geohash"] == geohash].shape)
    row = df[df["geohash"] == geohash].sort_values("time_minutes").iloc[-1]
    return row[[
        'demand_lag_15', 'demand_lag_hour', 'demand_lag_day',
        'demand_roll_mean_hour', 'demand_roll_std_hour',
    ]]

gp = GeohashProcessor.load("model/geohashsaver.pkl")

with open("model/traffic_model.pkl", "rb") as f:
    tmodel = pickle.load(f)
f.close()

with open("model/le_geohash.pkl", "rb") as f:
    le_geohash = pickle.load(f)
f.close()

with open("model/le_landmarks.pkl", "rb") as f:
    le_landmarks = pickle.load(f)
f.close()

with open("model/le_large_vehicles.pkl", "rb") as f:
    le_large_vehicles = pickle.load(f)
f.close()

with open("model/le_roadtype.pkl", "rb") as f:
    oh_roadtype = pickle.load(f)
f.close()

with open("model/le_weather.pkl", "rb") as f:
    oh_weather = pickle.load(f)
f.close()


class traffic(BaseModel):
    geohash: str
    day: int
    timestamp: str
    NumberofLanes: int
    RoadType: Literal['Residential', 'Street', 'Highway']
    LargeVehicles: Literal["Not Allowed", "Allowed"]
    Landmarks: str
    Temprature: float
    Weather: Literal['Sunny', 'Rainy', 'Foggy', 'Snowy']

@app.get("/")
def Home():
    return {"message": "welcome home"}

def run_prediction(state: traffic):
    user_input = state.model_dump()
    geohash = state.geohash
    
    lst = [
        'demand_lag_15', 'demand_lag_hour', 'demand_lag_day',
        'demand_roll_mean_hour', 'demand_roll_std_hour',
    ]

    user_df = pd.DataFrame([user_input])
    row = getlag(geohash)
    
    newdfcp = user_df.copy()
    newdfcp = time_change(newdfcp)
    newdfcp = labelEncode(newdfcp)
    newdfcp = gp.transform(newdfcp)
    newdfcp, *_ = preprocessT(newdfcp, le_geohash, le_large_vehicles, le_landmarks, oh_roadtype, oh_weather)
    newdfcp = newdfcp.reindex(columns=tmodel.feature_name_, fill_value=0)
    for col in lst:
        newdfcp[col] = row[col]

    prediction = tmodel.predict(newdfcp)
    return prediction



@app.post("/traffic/")
def trafic(state: traffic):
    prediction = run_prediction(state)
    print(prediction)
    return JSONResponse(status_code = 200, content={"Predicted_Demand": round(float(prediction[0]*100),2)})