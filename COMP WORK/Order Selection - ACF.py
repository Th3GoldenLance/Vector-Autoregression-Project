import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf #Autocorrelation and Partial Autocorrelation functions

# Paths for script and data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_folder = os.path.join(script_dir, 'CSV Data')
csv_file_path = os.path.join(csv_folder, 'Seasonally_Differenced_Data.csv')

data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True) #load the seasonally differenced data
data.index = data.index.to_period('Q') #set frequency of date index to quarterly to avoid ValueWarning in terminal
candidate_lags = [5, 6, 7, 8] #candidate lag lengths to analyze after selecting from AIC, BIC and LR-test
acf_pacf_results = [] #empty list to store numeric acf and pacf results

for lag in candidate_lags: #loop over each candidate lag
    print(f"\nAnalyzing residuals for model with lag length = {lag}...")
    model = VAR(data)
    fitted_model = model.fit(lag) #fit the VAR(lag) model
    residuals = fitted_model.resid #get residuals of fitted model

    for var in residuals.columns: #loop over each value to get ACF and PACF values to later store in csv
        acf_vals = acf(residuals[var], nlags=20)
        pacf_vals = pacf(residuals[var], nlags=20)

        #now, calculate the confidence intervals
        conf_int_acf = 1.96 / np.sqrt(len(residuals[var]))
        conf_int_pacf = 1.96 / np.sqrt(len(residuals[var]))

        for lag_index in range(len(acf_vals)): #append acf and pacf results to the list
            acf_pacf_results.append({
                'Variable': var,
                'Lag Length': lag,
                'Lag': lag_index,
                'ACF Value': acf_vals[lag_index],
                'ACF Conf Int Low': -conf_int_acf,
                'ACF Conf Int High': conf_int_acf,
                'PACF Value': pacf_vals[lag_index],
                'PACF Conf Int Low': -conf_int_pacf,
                'PACF Conf Int High': conf_int_pacf
            })


    #Now, plot ACF for each variable in the residuals
    num_vars = residuals.shape[1]
    cols = 2
    rows = 3

    fig_acf, axes_acf = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 12))
    axes_acf = axes_acf.flatten()

    for i, var in enumerate(residuals.columns):
        plot_acf(residuals[var], ax=axes_acf[i], title=f'ACF of Residuals - {var} (lag={lag})')

    if num_vars < len(axes_acf): #remove any unused axes (if any)
        for i in range(num_vars, len(axes_acf)):
            fig_acf.delaxes(axes_acf[i])

    fig_acf.tight_layout()
    plt.show()

    #Now, plot Partial ACF for each variable in the residuals
    fig_pacf, axes_pacf = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 12))
    axes_pacf = axes_pacf.flatten()

    for i, var in enumerate(residuals.columns):
        plot_pacf(residuals[var], ax=axes_pacf[i], title=f'PACF of Residuals - {var} (lag={lag})')

    
    if num_vars < len(axes_pacf): #remove any unused axes (if any)
        for i in range(num_vars, len(axes_pacf)):
            fig_acf.delaxes(axes_pacf[i])

    fig_pacf.tight_layout()
    plt.show()

acf_pacf_df = pd.DataFrame(acf_pacf_results) #create data frame from acf and pacf results
#save this to a csv file
acf_pacf_output_path = os.path.join(csv_folder, 'ACF_PACF_Values.csv')
acf_pacf_df.to_csv(acf_pacf_output_path, index=False)
print(f"ACF and PACF values saved to: {acf_pacf_output_path}") #notification