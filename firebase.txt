# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

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

# CSV Field Headers (includes warmup_ready flag)
CSV_HEADERS = [
    "timestamp", "ph", "tds", "ec", "water_temp", 
    "water_level", "flow_rate", "air_temp", "humidity", 
    "lux", "co2_ppm", "prediction", "confidence", "warmup_ready"
]

def save_locally(payload):
    """Appends payload data as a row in the local CSV file without overwriting."""
    file_exists = os.path.exists(CSV_FILE_PATH)
    try:
        with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
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

# Reference to the live node on Firebase
ref_latest = db.reference('/hydroponics/latest')

# ===============================
# SERIAL SETTINGS WITH AUTO-FALLBACK
# ===============================
BAUD_RATE = 115200
SERIAL_PORTS = ["/dev/ttyUSB0", "/dev/ttyACM0"]
ser = None

for port in SERIAL_PORTS:
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Connected to Serial Port: {port} at {BAUD_RATE} baud.")
        break
    except Exception:
        continue

if ser is None or not ser.is_open:
    print("Error: Could not open any serial port (/dev/ttyUSB0 or /dev/ttyACM0).")
    exit()

# ===============================
# LOAD ML MODEL
# ===============================
MODEL_PATH = "/home/pi/hydroponic_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"ML Model loaded successfully from {MODEL_PATH}!")
except Exception as e:
    print(f"Failed to load ML Model from {MODEL_PATH}: {e}")
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
# MAIN LOOP
# ===============================
print("===================================")
print("Hydroponic Realtime System Started")
print("===================================")

while True:
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            # Filter non-JSON startup messages from Arduino
            if not line or not (line.startswith('{') and line.endswith('}')):
                continue

            data = json.loads(line)

            # Safely parse raw sensor values from Arduino JSON
            ph           = float(data.get("ph", 7.0))
            tds          = float(data.get("tds", 0.0))
            ec           = float(data.get("ec", 0.0))
            water_temp   = float(data.get("water_temp", 25.0))
            air_temp     = float(data.get("air_temp", 25.0))
            humidity     = float(data.get("humidity", 50.0))
            water_level  = float(data.get("distance_cm", 0.0))
            flow_rate    = float(data.get("flow_rate", 0.0))
            lux          = float(data.get("lux", 0.0))
            co2_voltage  = float(data.get("co2_volt", 0.3))
            warmup_ready = bool(data.get("warmup_ready", False))

            # Run CO2 calculation & ML prediction
            co2_ppm = voltage_to_co2(co2_voltage)
            prediction, confidence = predict_status(
                ph, tds, ec, water_temp, humidity
            )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Structure payload
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
                "confidence": round(confidence, 2),
                "warmup_ready": warmup_ready
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

            print("\nSYSTEM STATUS")
            print("----------------------------")
            print("Warmup Ready  :", warmup_ready)
            print("Prediction    :", prediction)
            print("Confidence    :", f"{confidence:.2f}%")

            print("--> Realtime stream updated on Firebase & Local CSV!")
            print("===================================")

    except json.JSONDecodeError:
        continue

    except KeyboardInterrupt:
        print("\nProgram Stopped by User")
        if ser and ser.is_open:
            ser.close()
        break

    except Exception as e:
        print("Unexpected Error:", e)
        time.sleep(1)