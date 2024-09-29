import pandas as pd #For data manipulation and analysis
import numpy as np #For numerical operations
import matplotlib.pyplot as plt #For plotting graphs
import math #grid-size for visualizations
import seaborn as sns #For advanced data visualization

from statsmodels.tsa.stattools import adfuller, kpss #For stationarity tests: ADF and KPSS tests
from statsmodels.tsa.seasonal import STL #For seasonal decomposition
from statsmodels.graphics.tsaplots import plot_acf #For plotting autocorrelation and partial autocorrelation functions
import glob #For file pattern matching
import os

#os.path.abspath(__file__) gets the absolute path of the script file
#os.path.dirname() extracts the directory path from that absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))

csv_folder = os.path.join(script_dir, 'CSV Data') #os.path.join() combines the script directory with 'CSV Data' to form the path
csv_pattern = os.path.join(csv_folder, '*.csv') #This pattern will match all files ending with '.csv' in the 'CSV Data' folder
csv_files = glob.glob(csv_pattern) #glob.glob() returns a list of file paths that match the given pattern

# Verify that files are found
if not csv_files:
    print("No CSV files found in the specified directory.")
    print(f"Expected to find CSV files in: {csv_folder}")
else:
    print("CSV files found:")
    for file in csv_files:
        print(file)

# Initialize an empty list to store individual DataFrames
data_frames = []

# Proceed only if CSV files are found
if csv_files:
    for file in csv_files: #Loop through each CSV file and read it into a DataFrame
        var_name = os.path.splitext(os.path.basename(file))[0] #Extract variable name from file name
        df = pd.read_csv(file) #Read the CSV file
        df.rename(columns={'Value': var_name}, inplace=True) #Rename the 'Value' column to the variable name
        
        #Convert 'Date' column to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df.set_index('Date', inplace=True)
        
        data_frames.append(df) #Append the DataFrame to the list

    data = pd.concat(data_frames, axis=1) #Combines all DataFrames along the columns, aligning them on the 'Date' index
    
    data.sort_index(inplace=True)   #Sort the DataFrame by Date (index)
    
    #Check for missing values
    print("Missing values per variable:")
    print(data.isnull().sum())
    
    data.dropna(inplace=True) #Optionally, drop rows with any missing values
    
    variables = data.columns.tolist()  #Get the list of variable names
    
    #If these checks are cleared, then we can proceed with our analysis
else:
    print("Cannot proceed with analysis without data.")

#Plot time series of all variables for visual inspection of seasonality

variables = data.columns.tolist() #get list of the variables
num_vars = len(variables) #determine the no. of variables
cols = 2 #no. of columns for the subplot grid
rows = math.ceil(num_vars / cols) #3 rows in this case

fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize = (15, 5 * rows)) #create the figure and grid of subplots
axes = axes.flatten() #flatten axes for easy indexing

for i, var in enumerate(variables): #loop through each variable and corresponding axis
    ax = axes[i] #select the subplot axis
    ax.plot(data.index, data[var]) #plot the time series data for the variable on the selected axis
    #set title and axis labels for the plots
    ax.set_title(f'Time Series Plot of {var}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=45) #rotate x-axis labels if needed for better look

if num_vars < len(axes): #remove any unused subplots
    for i in range(num_vars, len(axes)):
        fig.delaxes(axes[i])

plt.tight_layout #adjust layout to prevent overlap
plt.show()

#Perform Seasonal Decomposition using Loess (STL)
for var in variables:
    series = data[var].dropna() #exclude missing values
    stl = STL(series, period=4) #period=4 for quarterly data
    result = stl.fit()
    result.plot() #plot the decomposed components (trend, seasonal, residual)
    plt.suptitle(f'STL Decomposition of {var}', fontsize = 16)
    plt.show()

#Visual inspection shows all series show cyclical behavior. Only in the 3mth Treasury Bill is the seasonality minimal compared to the trend
#Now, we seasonally difference the other 5 variables except for 3mth T-Bill

data_diff_seasonal = data.copy() #Create a copy of original data for seasonal differencing
vars_to_seasonally_diff = ['US CPI SA', 'US IP SA', 'US Unemployment SA', 'USDebt SA', 'USDXY SA'] #List of variables requiring seasonal differencing

for var in vars_to_seasonally_diff: #apply seasonal differencing with lag = 4 (quarterly data)
    data_diff_seasonal[var] = data[var].diff(periods=4) #update the data frame

data_diff_seasonal.dropna(inplace=True) #drop rows with NaN values from the differencing

print("Seasonally Differenced Data:") #Print the head of the seasonally differenced data to verify
print(data_diff_seasonal.head())

#Visualize the seasonally differenced series
num_vars_diff = len(vars_to_seasonally_diff) # number of variables
cols = 2 # number of columns for the subplot grid
rows = math.ceil(num_vars_diff / cols)

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, var in enumerate(vars_to_seasonally_diff):
    ax = axes[i]
    ax.plot(data_diff_seasonal.index, data_diff_seasonal[var])
    ax.set_title(f'Seasonally Differenced Series of {var}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Seasonally Differenced Value')
    ax.tick_params(axis='x', rotation=45)

if num_vars_diff < len(axes):
    for i in range(num_vars_diff, len(axes)):
        fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

data_diff_seasonal['3M TBill SA'] = data['3M TBill SA'] #include the non-differenced '3M TBill SA' series

#order the columns properly, placing 3M TBill SA first
final_columns_order = ['3M TBill SA'] + vars_to_seasonally_diff
data_diff_seasonal = data_diff_seasonal[final_columns_order]

#define output path to save data in the 'CSV Data' folder
csv_output_folder = os.path.join(script_dir, 'CSV Data')
output_path = os.path.join(csv_output_folder, 'Seasonally_Differenced_Data.csv')
data_diff_seasonal.to_csv(output_path) #Save the resulting data frame to the new CSV file

print(f"Seasonally differenced data (including '3M TBill SA') saved to: {output_path}") #confirm that the CSV was saved