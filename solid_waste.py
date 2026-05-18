import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Read CSV file
df = pd.read_csv("MinicipalGenerated.csv")

# Take all year columns
years = df.columns[1:]

X = []
Y = []

# Prepare data
for year in years:
    X.append([int(year)])
    Y.append(df[year].sum())

# Create model
model = LinearRegression()

# Train model
model.fit(X, Y)

# Predict existing values
predicted = model.predict(X)

# ---------------- ERROR CHECKING ----------------

# MAE
mae = mean_absolute_error(Y, predicted)

# MSE
mse = mean_squared_error(Y, predicted)

# RMSE
rmse = math.sqrt(mse)

# R2 Score
r2 = r2_score(Y, predicted)


# User input
future_year = int(input("Enter year to predict waste: "))

# Future prediction
prediction = model.predict([[future_year]])

# Print prediction
print("Predicted Solid Waste for", future_year, "=", prediction[0], "tons/year")

# Print errors
print("MAE =", mae)
print("MSE =", mse)
print("RMSE =", rmse)
print("R2 Score =", r2)
