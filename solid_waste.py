

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("MinicipalGenerated.csv")

years = df.columns[1:]

X = []
Y = []

for year in years:
    X.append([int(year)])
    Y.append(df[year].sum())

model = LinearRegression()

model.fit(X, Y)

predicted = model.predict(X)

future_year = int(input("Enter year to predict waste: "))

prediction = model.predict([[future_year]])
print("Predicted Solid Waste for", future_year, "=", prediction[0], "tons/year")
