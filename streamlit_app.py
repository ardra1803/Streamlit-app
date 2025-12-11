import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("AI-Powered Supply Chain Dashboard")

# ---------------- Load Data and Models ----------------
df = pd.read_csv("processed_data.csv")
delay_model = joblib.load("delay_model.pkl")
demand_model = joblib.load("demand_model.pkl")
scaler = joblib.load("scaler.pkl")
delay_feature_columns = joblib.load("delay_feature_columns.pkl")
demand_feature_columns = joblib.load("demand_feature_columns.pkl")

# ---------------- Key Metrics ----------------
st.header("Key Metrics")
st.metric("Total Assets", len(df))

# ---------------- Delay Prediction ----------------
features = df[delay_feature_columns]
features_scaled = scaler.transform(features)
df["Predicted_Delay"] = delay_model.predict(features_scaled)
st.metric("Predicted Delays", df["Predicted_Delay"].sum())

# Assets to restock
restock_df = df[df["Inventory_Level"] < df["Demand_Forecast"]]
st.metric("Assets to Restock", len(restock_df))

# ---------------- Shipment Delay Prediction ----------------
st.header("Shipment Delay Prediction")

latitude = st.number_input("Latitude", value=12.9716)
longitude = st.number_input("Longitude", value=77.5946)
inventory = st.number_input("Inventory Level", value=45)
temperature = st.number_input("Temperature", value=30)
humidity = st.number_input("Humidity", value=60)
waiting_time = st.number_input("Waiting Time", value=15)
user_transaction = st.number_input("User Transaction Amount", value=5000)
user_frequency = st.number_input("User Purchase Frequency", value=4)
asset_utilization = st.number_input("Asset Utilization", value=0.75)
demand_forecast = st.number_input("Demand Forecast", value=120)
delay_reason = st.number_input("Logistics Delay Reason (encoded)", value=3)
traffic_status = st.number_input("Traffic Status (encoded)", value=1)
asset_id = st.number_input("Asset ID (encoded)", value=10)

input_df = pd.DataFrame([{
    "Latitude": latitude,
    "Longitude": longitude,
    "Inventory_Level": inventory,
    "Temperature": temperature,
    "Humidity": humidity,
    "Waiting_Time": waiting_time,
    "User_Transaction_Amount": user_transaction,
    "User_Purchase_Frequency": user_frequency,
    "Asset_Utilization": asset_utilization,
    "Demand_Forecast": demand_forecast,
    "Logistics_Delay_Reason_enc": delay_reason,
    "Traffic_Status_enc": traffic_status,
    "Asset_ID_enc": asset_id
}])

# Predict Shipment Delay
input_delay = input_df[delay_feature_columns]
input_scaled = scaler.transform(input_delay)
delay_pred = delay_model.predict(input_scaled)[0]
delay_prob = delay_model.predict_proba(input_scaled)[0]

st.write("Predicted Logistics Delay:", delay_pred)
st.write("Probability of No Delay:", delay_prob[0])
st.write("Probability of Delay:", delay_prob[1])

# ---------------- Demand Forecast ----------------
st.header("Demand Forecast")

# Fill missing columns for XGBoost input
for col in demand_feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_demand = input_df[demand_feature_columns]
demand_pred = demand_model.predict(input_demand)[0]
st.write("Predicted Demand:", demand_pred)

# ---------------- Dashboard Tables ----------------
st.header("Inventory Overview")
st.dataframe(df[["Asset_ID_enc", "Inventory_Level", "Demand_Forecast"]])

st.header("Restocking Recommendations")
st.dataframe(restock_df[["Asset_ID_enc","Inventory_Level","Demand_Forecast"]])

st.header("Predicted Delays (Rerouting Recommendations)")
reroute_df = df[df["Predicted_Delay"] == 1]
st.dataframe(reroute_df[["Asset_ID_enc","Latitude","Longitude","Inventory_Level"]])

# ---------------- Visualizations ----------------
st.header("Logistics Trends")
st.subheader("Inventory vs Forecast")
st.line_chart(df[["Inventory_Level","Demand_Forecast"]])

st.subheader("Predicted Delays Distribution")
st.bar_chart(df["Predicted_Delay"].value_counts())

