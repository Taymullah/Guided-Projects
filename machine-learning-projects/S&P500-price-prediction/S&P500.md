# Context

In this guided project, we're specifically focusing on predicting whether the S&P 500 index will close higher or lower the next day. By leveraging historical stock data, feature engineering techniques, and Scikit-Learn's Random Forest Classifier, the project demonstrates the process of building a predictive model from data collection to evaluation.

### Key Features

- **Data Collection:** Fetches historical S&P 500 data using the `yfinance` library.
- **Data Preprocessing:** Cleans the dataset by removing unnecessary columns and creating target variables.
- **Feature Engineering:** Generates new predictors based on rolling averages and trend indicators.
- **Model Training:** Utilizes Random Forest Classifier for predicting stock movements.
- **Backtesting:** Implements a backtesting mechanism to simulate model performance over time.
- **Evaluation:** Assesses model performance using precision scores and prediction distribution.
