import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_folder = os.path.join(script_dir, 'CSV Data')
csv_file_path = os.path.join(csv_folder, 'Seasonally_Differenced_Data.csv')
data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True) #load the seasonally differenced data
lag_lengths = [6,7,8] #lag length of models to consider

irf_periods = 20 #no. of periods for Impulse Response Function to trace the effect of shocks to the system

def generate_irf(data,lag_length,periods): #function to fit VAR model and generate IRFs
    model = VAR(data)
    fitted_model = model.fit(lag_length)
    irf = fitted_model.irf(periods)
    return irf, fitted_model

for lag_length in lag_lengths: #loop over each lag to generate IRFs
    irf, fitted_model = generate_irf(data, lag_length, irf_periods)
    irf_df = pd.DataFrame() #store IRF data for all variables and shocks

    variables = data.columns.tolist() #get variable names from column headees
    for i, shock_var in enumerate(variables): #loop thorugh each combination of shock and response variable
        for j, response_var in enumerate(variables):
            #Extract IRF values for given pair of shock and response var
            irf_values = irf.irfs[:, j, i]  # response of `response_var` to `shock_var`
            stderr_values = irf.stderr(orth=False)[:, j, i]  # standard errors for confidence intervals

            # Calculate lower and upper confidence bounds (assuming 95% confidence interval)
            lower_conf = irf_values - 1.96 * stderr_values
            upper_conf = irf_values + 1.96 * stderr_values

            df_pair = pd.DataFrame({
                'Period': range(irf_periods + 1),  # include all periods
                'IRF': irf_values,
                'Lower Conf': lower_conf,
                'Upper Conf': upper_conf,
                'Shock Variable': shock_var,
                'Response Variable': response_var
            })
            irf_df = pd.concat([irf_df, df_pair], ignore_index=True) #Apply to the main IRF Data Frame

    output_folder = os.path.join(script_dir, 'CSV Data')
    output_file = os.path.join(output_folder, f'IRF_Lag_{lag_length}.csv')
    irf_df.to_csv(output_file, index=False) #save data frame to the csv
    print(f"IRF data for lag length {lag_length} saved to: {output_file}")
    
    #Plot the IRF for each variable's shock
    fig = irf.plot(orth=False)
    fig.set_size_inches(14, 10)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.suptitle(f'Impulse Response Functions (Lag Length = {lag_length})', fontsize=16)

    plt.show()

    #Plot cumulative IRF plots
    fig_cum = irf.plot_cum_effects(orth=False)
    fig_cum.set_size_inches(14,10)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig_cum.suptitle(f'Cumulative IRFs (Lag Length = {lag_length})', fontsize=16)

    plt.show()