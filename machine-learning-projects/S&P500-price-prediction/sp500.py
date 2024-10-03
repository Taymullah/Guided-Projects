import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier

# S&P 500 ticker
sp500 = yf.Ticker("^GSPC")
# Get historical data
sp500_data = sp500.history(period="max")

# Remove unnecessary columns
del sp500_data["Dividends"]
del sp500_data["Stock Splits"]

# Create target column
sp500_data["Tomorrow"] = sp500_data["Close"].shift(-1)
sp500_data["Target"] = np.where(sp500_data["Tomorrow"] > sp500_data["Close"], 1, 0)
sp500_data = sp500_data.loc["1990-01-01":].copy()

# Define predictors
predictors = ["Close", "High", "Low", "Open", "Volume"]


# Function to predict
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# Function to backtest
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[:i].copy()
        test = data.iloc[i : (i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# Initial model training and backtesting
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
predictions = backtest(sp500_data, model, predictors)

# Evaluate predictions
print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))

# Adding new predictors based on rolling averages and trends
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500_data.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500_data[ratio_column] = sp500_data["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    sp500_data[trend_column] = sp500_data.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

sp500_data = sp500_data.dropna()

# Train and backtest with new predictors
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
predictions = backtest(sp500_data, model, new_predictors)

# Evaluate new predictions
print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))
