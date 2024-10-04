import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from 'forecast_history.csv'
data = pd.read_csv('forecast_history.csv')

# Rename 'Unnamed: 0' to 'Year'
data.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)

# Display the original data
print("Original Data:")
print(data)

# Data Cleaning
# Correct typos and handle inconsistencies
data.replace({'I5%': '15%', '760O00': '760000', '34$': '34%', '73000': '730000'}, inplace=True)
data['Median house price'] = pd.to_numeric(data['Median house price'], errors='coerce')

# Remove percentage signs and convert forecasts to numeric values
def clean_forecast(value):
    if pd.isnull(value):
        return np.nan
    else:
        value = str(value).strip().replace('%', '')
        try:
            return float(value)
        except ValueError:
            return np.nan

data['Westpac: 4 year forecast'] = data['Westpac: 4 year forecast'].apply(clean_forecast)
data['Joe Bloggs: 2 year forecast'] = data['Joe Bloggs: 2 year forecast'].apply(clean_forecast)
data['Harry Spent: 5 year forecast'] = data['Harry Spent: 5 year forecast'].apply(clean_forecast)

# Display the cleaned data
print("\nCleaned Data:")
print(data)

# Forecast Calculation Functions
def calculate_forecasted_price(base_price, forecast_percent):
    if pd.isnull(base_price) or pd.isnull(forecast_percent):
        return np.nan
    else:
        return base_price * (1 + forecast_percent / 100)

# Calculate forecasted prices for each forecaster
data['Westpac_predicted'] = data.apply(
    lambda row: calculate_forecasted_price(row['Median house price'], row['Westpac: 4 year forecast']), axis=1)

data['JoeBloggs_predicted'] = data.apply(
    lambda row: calculate_forecasted_price(row['Median house price'], row['Joe Bloggs: 2 year forecast']), axis=1)

data['HarrySpent_predicted'] = data.apply(
    lambda row: calculate_forecasted_price(row['Median house price'], row['Harry Spent: 5 year forecast']), axis=1)

# Calculate Mean Absolute Percentage Error (MAPE) for each forecaster
def mean_absolute_percentage_error(y_true, y_pred):
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (y_true != 0)
    y_true, y_pred = y_true[mask], y_pred[mask]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

westpac_mape = mean_absolute_percentage_error(data['Median house price'], data['Westpac_predicted'])
joe_bloggs_mape = mean_absolute_percentage_error(data['Median house price'], data['JoeBloggs_predicted'])
harry_spent_mape = mean_absolute_percentage_error(data['Median house price'], data['HarrySpent_predicted'])

# Print MAPE results
print("\nMean Absolute Percentage Error (MAPE):")
print(f"Westpac MAPE: {westpac_mape:.2f}%")
print(f"Joe Bloggs MAPE: {joe_bloggs_mape:.2f}%")
print(f"Harry Spent MAPE: {harry_spent_mape:.2f}%")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(data['Year'], data['Median house price'], label='Actual Median House Price', marker='o')

# Plot forecasts
plt.plot(data['Year'], data['Westpac_predicted'], label='Westpac Prediction', marker='x')
plt.plot(data['Year'], data['JoeBloggs_predicted'], label='Joe Bloggs Prediction', marker='^')
plt.plot(data['Year'], data['HarrySpent_predicted'], label='Harry Spent Prediction', marker='s')

plt.xlabel('Year')
plt.ylabel('House Price')
plt.title('Actual vs Predicted Median House Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as an image file
plt.savefig('median_house_price_forecasts.png')

# Display the plot
plt.show()
