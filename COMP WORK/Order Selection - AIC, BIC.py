import pandas as pd
import os
from statsmodels.tsa.api import VAR

# Paths for script and data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_folder = os.path.join(script_dir, 'CSV Data')
csv_file_path = os.path.join(csv_folder, 'Seasonally_Differenced_Data.csv')

data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True) #load the seasonally differenced data
data.index = data.index.to_period('Q') #Set frequency of Date index to quarterly to avoid any ValueWarning in the output

#set the range of lag lengths to test
max_lags = 15 #4 years of data
lag_range = range(1, max_lags + 1) #range is exclusive so we use max_lags + 1

#initialize lists to store AIC and BIC for each lag length
aic_values = []
bic_values = []

#loop over each lag length and fit the VAR(p) model
for lag in lag_range:
    model = VAR(data)
    fitted_model = model.fit(lag)
    aic_values.append(fitted_model.aic)
    bic_values.append(fitted_model.bic)
    print(f"Lag {lag}: AIC = {fitted_model.aic}, BIC = {fitted_model.bic}")

results_df = pd.DataFrame({'Lag Length': lag_range, 'AIC': aic_values, 'BIC': bic_values}) #create data frame to display AIC and BIC for each lag length

#Display the tableprint("\nAIC and BIC values for different lag lengths:")
print(results_df)

#Save the table to a CSV File
output_path = os.path.join(csv_folder, 'AIC_BIC_Comparison.csv')
results_df.to_csv(output_path, index=False)
print(f"AIC and BIC comparison saved to: {output_path}")