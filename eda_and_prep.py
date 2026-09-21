import pandas as pd

df = pd.read_csv("Nassau Candy Distributor.csv")

print("--- Raw Sample Dates ---")
print(df[['Order ID', 'Order Date', 'Ship Date']].head(10))

print("\n--- Date String Formats Check ---")
print(f"Order Date sample: {df['Order Date'].iloc[0]}")
print(f"Ship Date sample:  {df['Ship Date'].iloc[0]}")