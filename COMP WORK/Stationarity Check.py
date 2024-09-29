import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller, kpss

#path for the scripts and data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_folder = os.path.join(script_dir, 'CSV Data')
csv_file_path = os.path.join(csv_folder, 'Seasonally_Differenced_Data.csv')

data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True) #Load the seasonally differenced data
variables = data.columns.tolist() #variable names that we want to check for stationarity

#function for performing stationary tests
def check_stationarity(series, var_name):
    #perform the Augmented Dickey-Fuller test (ADF)
    adf_result = adfuller(series.dropna()) #drop na values
    print(f'{var_name} - ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}')
    if adf_result[1] < 0.05:
        print(f"{var_name} is stationary according to the ADF test (p < 0.05).")
    else:
        print(f"{var_name} is not stationary according to the ADF test (p >= 0.05).")
    #we're testing the null hypothesis that the series has a unit root i.e. it is non-stationary
    #if the p-value is less than 0.05, then the series is considered stationary

    #Perform the Kwiatkowski-Phillips-Schmidt-Shin Test (KPSS) test
    kpss_result = kpss(series.dropna(), regression='c') #drop na values
    print(f'{var_name} - KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')
    if kpss_result[1] < 0.05:
        print(f"{var_name} is not stationary according to the KPSS test (p < 0.05).")
    else:
        print(f"{var_name} is stationary according to the KPSS test (p >= 0.05).")

    print("-" * 50)
    #we're testing the null hypothesis that the series is stationary
    #If the p-value is less than 0.05, then the series is considered non-stationary

for var in variables: #perform the two stationarity tests on all the variables
    print(f"Testing for stationarity in {var}:")
    check_stationarity(data[var], var)