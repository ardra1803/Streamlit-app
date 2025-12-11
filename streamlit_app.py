import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
import os

st.title("AI-Powered Supply Chain Dashboard (Debug Mode Enabled)")

# ------------------------------------------------------------
# DEBUG BLOCK → Shows errors instead of blank screen
# ------------------------------------------------------------
st.subheader("Debug Info")

try:
    st.write("📁 Files in working directory:", os.listdir())

    # Try loading data
    st.write("Loading processed_data.csv ...")
    df = pd.read_csv("processed_data.csv")
    st.success("processed_data.csv loaded successfully!")

    # Try loading models
    st.write("Loading delay_model.pkl ...")
    delay_model = joblib.load("delay_model.pkl")
    st.success("delay_model.pkl loaded!")

    st.write("Loading demand_model.pkl ...")
    demand_model = joblib.load("demand_model.pkl")
    st.success("demand_model.pkl loaded!")

    st.write("Loading scaler.pkl ...")
    scaler = joblib.load("scaler.pkl")
    st.success("scaler.pkl loaded!")

except Exception as e:
    st.error("❌ Error while loading models/data")
    st.code(str(e))
    st.code(traceback.format_exc())
    st.stop()

# ------------------------------------------------------------
# MAIN APP (Only runs if all files loaded successfully)
# ------------------------------------------------------------

st.header("Key Metrics")
st.metric("Total Assets", len(df))

# Predict delays
try:
    feature_cols = [
        "Latitude","Longitude","Inventory_Level","Logistics_Delay_Reason_enc",
        "Traffic_Status_enc","Asset_ID_enc","Waiting_Time","Temperature","Humidity",
        "User_Transaction_Amount","User_Purchase_Frequency","Asset_Utilization","Demand_Forecast"
    ]

    features = df[feature_cols]
    features_scaled = scaler.transform(features)

    df["Predicted_Delay"] = delay_model.predict(features_scaled)
    st.metric("Predicted Delays", df["Predicted_Delay"].sum())

except Exception as e:
    st.error("❌ Error in delay prediction block")
    st.code(traceback.format_exc())
    st.stop()


# Assets to restock
restock_df = df[df["Inventory_Level"] < df["Demand_Forecast"]]
st.metric("Assets to Restock", len(restock_df))

# ---------------- Shipment Delay Section ----------------
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

try:
    input_scaled = scaler.transform(input_df)
    delay_pred = delay_model.predict(input_scaled)[0]
    delay_prob = delay_model.predict_proba(input_scaled)[0]

    st.write("Predicted Logistics Delay:", delay_pred)
    st.write("Probability of No Delay:", delay_prob[0])
    st.write("Probability of Delay:", delay_prob[1])

except Exception as e:
    st.error("❌ Error in shipment prediction block")
    st.code(traceback.format_exc())

# ---------------- Demand Forecast Block ----------------
st.header("Demand Forecast")

try:
    demand_pred = demand_model.predict(input_df)[0]
    st.write("Predicted Demand:", demand_pred)
except Exception as e:
    st.error("❌ Error in demand forecast block")
    st.code(traceback.format_exc())

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
