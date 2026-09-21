import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("1. Loading raw dataset...")
df = pd.read_csv("Nassau Candy Distributor.csv")

# Factory coordinates mapping
factory_coords = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107)
}

# Product-to-factory static assignment
factory_map = {
    'Wonka Bar - Nutty Crunch Surprise': "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows': "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious': "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate': "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel': "Wicked Choccy's",
    'Laffy Taffy': "Sugar Shack",
    'SweeTARTS': "Sugar Shack",
    'Nerds': "Sugar Shack",
    'Fun Dip': "Sugar Shack",
    'Fizzy Lifting Drinks': "Sugar Shack",
    'Everlasting Gobstopper': "Secret Factory",
    'Hair Toffee': "The Other Factory",
    'Lickable Wallpaper': "Secret Factory",
    'Wonka Gum': "Secret Factory",
    'Kazookles': "The Other Factory"
}

df['Factory'] = df['Product Name'].map(factory_map)
df['Factory_Lat'] = df['Factory'].map(lambda x: factory_coords[x][0])
df['Factory_Lon'] = df['Factory'].map(lambda x: factory_coords[x][1])

# Regional approximate coordinates for transit distance estimation
region_coords = {
    'Atlantic': (37.0, -78.0),
    'Gulf': (30.5, -90.0),
    'Interior': (39.5, -98.0),
    'Pacific': (37.5, -120.0)
}

df['Dest_Lat'] = df['Region'].map(lambda r: region_coords.get(r, (38.0, -95.0))[0])
df['Dest_Lon'] = df['Region'].map(lambda r: region_coords.get(r, (38.0, -95.0))[1])

# Euclidean route distance proxy (in degrees)
df['Distance_Proxy'] = np.sqrt(
    (df['Dest_Lat'] - df['Factory_Lat'])**2 + 
    (df['Dest_Lon'] - df['Factory_Lon'])**2
)

# Standard lead time baseline based on Ship Mode (days)
mode_base = {
    'Same Day': 0.5,
    'First Class': 2.0,
    'Second Class': 3.5,
    'Standard Class': 5.0
}

# Generate realistic Lead Time (Days): Base mode time + distance delay + noise
np.random.seed(42)
df['Lead Time (Days)'] = (
    df['Ship Mode'].map(mode_base) + 
    (df['Distance_Proxy'] * 0.15) + 
    np.random.normal(0, 0.3, len(df))
).round(1)

# Ensure no non-positive lead times
df['Lead Time (Days)'] = df['Lead Time (Days)'].clip(lower=0.5)

print("\n2. Feature engineering complete. Summary stats for Lead Time (Days):")
print(df['Lead Time (Days)'].describe())

# 3. Model Training
features = ['Factory', 'Region', 'Ship Mode', 'Distance_Proxy']
X = pd.get_dummies(df[features])
y = df['Lead Time (Days)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n3. Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"-> Mean Absolute Error: {mae:.2f} days")
print(f"-> R2 Score: {r2:.4f}")

# 4. Save artifacts
joblib.dump(model, 'lead_time_model.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')
df.to_csv('cleaned_nassau_candy.csv', index=False)

print("\n-> Clean dataset saved as 'cleaned_nassau_candy.csv'")
print("-> Model artifacts saved successfully!")