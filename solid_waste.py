# Simple Solid Waste Prediction Project

import pandas as pd
from sklearn.linear_model import LinearRegression

# Read CSV file
df = pd.read_csv("MinicipalGenerated.csv")

# Take all year columns automatically
years = df.columns[1:]

X = []
Y = []

# Create input and output data
for year in years:
    X.append([int(year)])      # Input year
    Y.append(df[year].sum())  # Total waste of all districts

# Create model
model = LinearRegression()

# Train model
model.fit(X, Y)

# Take year input from user
future_year = int(input("Enter year to predict waste: "))

# Predict future waste
prediction = model.predict([[future_year]])

# Print result
print("Predicted Solid Waste for", future_year, "=", prediction[0], "tons/year")