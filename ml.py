# -*- coding: utf-8 -*-

import serial
import json
import time
import math
import joblib

# ===============================
# SETTINGS
# ===============================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

# ===============================
# LOAD ML MODEL
# ===============================
model = joblib.load("/home/pi/hydroponic_model.pkl")

# ===============================
# MG811 CO2 SETTINGS
# ===============================
V400_VOLT = 0.320
V1000_VOLT = 0.270

slope = (math.log10(1000) - math.log10(400)) / (V1000_VOLT - V400_VOLT)

# ===============================
# CO2 Conversion
# ===============================
def voltage_to_co2(voltage):

    logppm = math.log10(400) + (voltage - V400_VOLT) * slope
    ppm = pow(10, logppm)

    if ppm < 400:
        ppm = 400

    return int(ppm)

# ===============================
# ML Prediction
# ===============================
def predict_status(ph, ec, water_temp,
                   air_temp, humidity,
                   water_level):

    sample = [[
        float(ph),
        float(ec),
        float(water_temp),
        float(air_temp),
        float(humidity),
        float(water_level)
    ]]

    prediction = model.predict(sample)[0]
    confidence = model.predict_proba(sample).max() * 100

    return prediction, confidence

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
    print("Cannot open serial port")
    print(e)
    exit()

print("===================================")
print("Hydroponic AI Started")
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

        ph = float(data["ph"])
        tds = float(data["tds"])
        ec = float(data["ec"])

        water_temp = float(data["water_temp"])

        air_temp = float(data["air_temp"])
        humidity = float(data["humidity"])

        water_level = float(data["distance_cm"])

        flow_rate = float(data["flow_rate"])

        lux = float(data["lux"])

        co2_voltage = float(data["co2_volt"])

        co2_ppm = voltage_to_co2(co2_voltage)

        prediction, confidence = predict_status(
            ph,
            ec,
            water_temp,
            air_temp,
            humidity,
            water_level
        )

        print("\n===================================")
        print("HYDROPONIC MONITOR")
        print("===================================")

        print("\nWATER")
        print("----------------------------")
        print("pH           :", round(ph, 2))
        print("TDS          :", round(tds, 1), "ppm")
        print("EC           :", round(ec, 2), "mS/cm")
        print("Water Temp   :", round(water_temp, 1), "C")
        print("Water Level  :", round(water_level, 1), "cm")
        print("Flow Rate    :", round(flow_rate, 2), "L/min")

        print("\nAIR")
        print("----------------------------")
        print("Air Temp     :", round(air_temp, 1), "C")
        print("Humidity     :", round(humidity, 1), "%")
        print("Light        :", round(lux, 1), "Lux")
        print("CO2          :", co2_ppm, "ppm")

        print("\nAI RESULT")
        print("----------------------------")
        print("Prediction   :", prediction)
        print("Confidence   :", "{:.2f}".format(confidence), "%")

        print("===================================")

    except json.JSONDecodeError:
        continue

    except KeyboardInterrupt:
        print("\nProgram Stopped")
        break

    except Exception as e:
        print("Error:", e)