# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# === Paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "car data.csv")
model_path = os.path.join(current_dir, "car_price_model.pkl")

# === Load the data ===
df = pd.read_csv(data_path)

# === Preprocessing ===
df['Car_Age'] = 2025 - df['Year']  # Calculate car age
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)  # Drop unused columns

# === Manually encode categorical features ===
fuel_map = {'Petrol': 2, 'Diesel': 1, 'CNG': 0}
seller_map = {'Dealer': 0, 'Individual': 1}
trans_map = {'Manual': 1, 'Automatic': 0}

df['Fuel_Type'] = df['Fuel_Type'].map(fuel_map)
df['Seller_Type'] = df['Seller_Type'].map(seller_map)
df['Transmission'] = df['Transmission'].map(trans_map)

# === Features and target ===
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Train model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Save the model ===
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")