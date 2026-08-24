# -*- coding: utf-8 -*-

import serial
import json
import time
import math
import joblib
import pandas as pd
import csv
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

# ===============================
# LOCAL CSV STORAGE SETUP
# ===============================
CSV_FILE_PATH = "/home/pi/hydroponic_history.csv"

# CSV Field Headers
CSV_HEADERS = [
    "timestamp", "ph", "tds", "ec", "water_temp", 
    "water_level", "flow_rate", "air_temp", "humidity", 
    "lux", "co2_ppm", "prediction", "confidence"
]

def save_locally(payload):
    """Appends payload data as a row in the local CSV file without overwriting."""
    file_exists = os.path.exists(CSV_FILE_PATH)
    
    try:
        # mode='a' opens the file in APPEND mode (never resets existing data)
        with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
            
            # If the file is newly created, write header row first
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(payload)
            print("--> Data appended to local CSV successfully!")
    except Exception as e:
        print("Local CSV save error:", e)


# ===============================
# FIREBASE SETUP
# ===============================
CRED_PATH = "/home/pi/firebase_key.json"
DATABASE_URL = "https://charging-station-f1f46-default-rtdb.asia-southeast1.firebasedatabase.app/"

try:
    cred = credentials.Certificate(CRED_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': DATABASE_URL
    })
    print("Firebase connected successfully!")
except Exception as e:
    print("Firebase initialization failed:", e)
    exit()

# Reference to the single live node
ref_latest = db.reference('/hydroponics/latest')

# ===============================
# SERIAL SETTINGS
# ===============================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

# ===============================
# LOAD ML MODEL
# ===============================
try:
    model = joblib.load("/home/pi/hydroponic_model.pkl")
    print("ML Model loaded successfully!")
except Exception as e:
    print("Failed to load ML Model:", e)
    exit()

# ===============================
# MG811 CO2 SETTINGS & HELPER
# ===============================
V400_VOLT = 0.320
V1000_VOLT = 0.270
slope = (math.log10(1000) - math.log10(400)) / (V1000_VOLT - V400_VOLT)

def voltage_to_co2(voltage):
    logppm = math.log10(400) + (voltage - V400_VOLT) * slope
    ppm = pow(10, logppm)
    if ppm < 400:
        ppm = 400
    return int(ppm)

# ===============================
# ML PREDICTION HELPER (5 Features)
# ===============================
def predict_status(ph, tds, ec, water_temp, humidity):
    sample = pd.DataFrame([{
        'pH': float(ph),
        'TDS': float(tds),
        'EC_mS_cm': float(ec),
        'water_temp': float(water_temp),
        'DHT_humidity': float(humidity)
    }])

    prediction = model.predict(sample)[0]
    confidence = model.predict_proba(sample).max() * 100

    return str(prediction), float(confidence)

# ===============================
# OPEN SERIAL PORT
# ===============================
try:
    ser = serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=1
    )
    time.sleep(2)
except Exception as e:
    print("Cannot open serial port:", e)
    exit()

print("===================================")
print("Hydroponic Realtime System Started")
print("===================================")

# ===============================
# MAIN LOOP
# ===============================
while True:
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if line == "":
            continue

        data = json.loads(line)

        # Parse raw sensor values from Arduino JSON
        ph          = float(data["ph"])
        tds         = float(data["tds"])
        ec          = float(data["ec"])
        water_temp  = float(data["water_temp"])
        air_temp    = float(data["air_temp"])
        humidity    = float(data["humidity"])
        water_level = float(data["distance_cm"])
        flow_rate   = float(data["flow_rate"])
        lux         = float(data["lux"])
        co2_voltage = float(data["co2_volt"])

        # Run CO2 calculation & ML prediction
        co2_ppm = voltage_to_co2(co2_voltage)
        prediction, confidence = predict_status(
            ph, tds, ec, water_temp, humidity
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Payload structure for Firebase & Local Storage
        payload = {
            "timestamp": timestamp,
            "ph": round(ph, 2),
            "tds": round(tds, 1),
            "ec": round(ec, 2),
            "water_temp": round(water_temp, 1),
            "water_level": round(water_level, 1),
            "flow_rate": round(flow_rate, 2),
            "air_temp": round(air_temp, 1),
            "humidity": round(humidity, 1),
            "lux": round(lux, 1),
            "co2_ppm": co2_ppm,
            "prediction": prediction,
            "confidence": round(confidence, 2)
        }

        # ---------------- FIREBASE REALTIME UPDATE ----------------
        ref_latest.set(payload)

        # ---------------- LOCAL CSV APPEND ----------------
        save_locally(payload)

        # ---------------- CONSOLE DISPLAY ----------------
        print("\n===================================")
        print(f"HYDROPONIC MONITOR [{timestamp}]")
        print("===================================")

        print("\nWATER")
        print("----------------------------")
        print("pH            :", round(ph, 2))
        print("TDS           :", round(tds, 1), "ppm")
        print("EC            :", round(ec, 2), "mS/cm")
        print("Water Temp    :", round(water_temp, 1), "C")
        print("Water Level   :", round(water_level, 1), "cm")
        print("Flow Rate     :", round(flow_rate, 2), "L/min")

        print("\nAIR")
        print("----------------------------")
        print("Air Temp      :", round(air_temp, 1), "C")
        print("Humidity      :", round(humidity, 1), "%")
        print("Light         :", round(lux, 1), "Lux")
        print("CO2           :", co2_ppm, "ppm")

        print("\nAI RESULT")
        print("----------------------------")
        print("Prediction    :", prediction)
        print("Confidence    :", "{:.2f}".format(confidence), "%")

        print("--> Realtime stream updated on Firebase!")
        print("===================================")

    except json.JSONDecodeError:
        continue

    except KeyboardInterrupt:
        print("\nProgram Stopped")
        if 'ser' in locals() and ser.is_open:
            ser.close()
        break

    except Exception as e:
        print("Error:", e)