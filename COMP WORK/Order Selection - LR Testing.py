import pandas as pd
import os
from statsmodels.tsa.api import VAR
from scipy.stats import chi2 #for chi-sq test statistic

# Paths for script and data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_folder = os.path.join(script_dir, 'CSV Data')
csv_file_path = os.path.join(csv_folder, 'Seasonally_Differenced_Data.csv')

data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True) #load seasonally differenced data
data.index = data.index.to_period('Q') #Set the frequency of the Date index to quarterly to avoid value warnings in the terminal

max_lags = 15 #define max lag length to test
K = data.shape[1] #set K as the no. of columns in the series data i.e. no. of variables in the series
lr_results = [] #empty list to store result of likelihood ratio tests

for lag in range(1, max_lags + 1): #loop over lag length 1 to max_lags (=15)
    #Fit VAR models with p and p-1 lags
    model_p1 = VAR(data).fit(lag)
    model_p0 = VAR(data).fit(lag - 1)

    #calculate log-likelihoods for chi-sq test statistic
    ll_p1 = model_p1.llf  # Log-likelihood of model with lag p
    ll_p0 = model_p0.llf  # Log-likelihood of model with lag p-1

    lr_stat = 2 * (ll_p1 - ll_p0) #calculate chi-sq test statistic (or LR test stat)
    df = K**2 #set degrees of freedom = K^2
    p_value = chi2.sf(lr_stat, df) #calculate p-value given d.f and test-stat

    #store test result in a dictionary
    lr_results.append({
        'Lag p-1': lag - 1,
        'Lag p': lag,
        'LR Statistic': lr_stat,
        'p-value': p_value,
        'Degrees of Freedom': df
    })


lr_df = pd.DataFrame(lr_results) #Convert results into a data frame

#display results
print("\nLikelihood Ratio Test Results:")
print(lr_df)

#Save results into a CSV
output_path = os.path.join(csv_folder, 'LR_Test_Results.csv')
lr_df.to_csv(output_path, index=False)
print(f"LR test results saved to: {output_path}")