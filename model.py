import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestClassifier

# Suppress version warnings during training
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -------------------------------------------------------------
# 1. Rule Logic matching Hydroponic Thresholds
# (TDS up to 1050 PPM / EC up to 2.1 mS/cm considered Good)
# -------------------------------------------------------------
def apply_rules(row):
    ph = row['pH']
    ec = row['EC_mS_cm']
    hum = row['DHT_humidity']
    wtemp = row['water_temp']
    
    # Critical condition boundaries
    if (ph < 5.3 or ph > 7.0) or (ec < 0.8 or ec > 2.5) or (hum < 50 or hum > 90) or (wtemp >= 36.0):
        return 'Critical'
        
    # Optimum condition boundaries (EC up to 2.1 mS/cm = 1050 PPM TDS)
    if (5.8 <= ph <= 6.5) and (1.2 <= ec <= 2.1) and (60 <= hum <= 80) and (15.0 <= wtemp <= 32.0):
        return 'Good'
        
    return 'Warning'

# -------------------------------------------------------------
# 2. Synthetic Boundary Dataset Generation
# -------------------------------------------------------------
np.random.seed(42)

# Synthetic 'Good' dataset (EC range 1.2 to 2.1 -> TDS 600 to 1050 PPM)
good_ec = np.random.uniform(1.2, 2.1, 5000)
synth_good = pd.DataFrame({
    'pH': np.random.uniform(5.8, 6.5, 5000),
    'EC_mS_cm': good_ec,
    'TDS': good_ec * 500.0,  # Matches ecValue = tdsValue / 500.0 from Arduino
    'water_temp': np.random.uniform(15.0, 32.0, 5000),
    'DHT_humidity': np.random.uniform(60.0, 80.0, 5000)
})

# Synthetic 'Warning' / 'Critical' dataset
critical_ec = np.random.uniform(0.1, 4.0, 10000)
synth_critical = pd.DataFrame({
    'pH': np.random.uniform(0.0, 14.0, 10000),
    'EC_mS_cm': critical_ec,
    'TDS': critical_ec * 500.0,  # Matches Arduino conversion factor
    'water_temp': np.random.uniform(5.0, 45.0, 10000),
    'DHT_humidity': np.random.uniform(20.0, 95.0, 10000)
})

# Feature list matching Arduino JSON keys used in firebase.py
features = ['pH', 'TDS', 'EC_mS_cm', 'water_temp', 'DHT_humidity']

# Load desktop CSV if present; fallback to synthetic if missing
csv_path = '/home/pi/Desktop/IoTData_With_EC_Final_Labeled.csv'
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'TDS' not in df.columns and 'EC_mS_cm' in df.columns:
            df['TDS'] = df['EC_mS_cm'] * 500.0
            
        all_data = pd.concat([df[features], synth_good, synth_critical], ignore_index=True)
        print(f"Loaded existing dataset from {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file, falling back to synthetic dataset: {e}")
        all_data = pd.concat([synth_good, synth_critical], ignore_index=True)
else:
    print(f"File {csv_path} not found. Training model using synthetic boundary dataset...")
    all_data = pd.concat([synth_good, synth_critical], ignore_index=True)

# Apply labeling rules across entire dataset
all_data['Label'] = all_data.apply(apply_rules, axis=1)

# -------------------------------------------------------------
# 3. Model Training & Export
# -------------------------------------------------------------
X = all_data[features]
y = all_data['Label']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

model_output_path = '/home/pi/hydroponic_model.pkl'
joblib.dump(model, model_output_path)

print(f"Model successfully retrained and saved to {model_output_path}!")