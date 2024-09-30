import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

# Main directory and file paths
main_directory = os.path.dirname(os.path.abspath(__file__))
csv_subdirectory = os.path.join(main_directory, 'CSV Data')

# Define the CSV file to analyze
csv_file = 'US Debt SA.csv'  # Adjust as needed for other time series
csv_path = os.path.join(csv_subdirectory, csv_file)

# Read the time series data
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)

# Perform STL decomposition
series_name = df.columns[0]
series = df[series_name].dropna()
stl = STL(series, period=4)
result = stl.fit()

# Plot the STL decomposition
fig = result.plot()
fig.set_size_inches(10, 8)

# Access all axes from the generated figure
for ax in fig.axes:
    # Set major ticks for every 4 years
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Set minor ticks for each quarter without labels
    ax.xaxis.set_minor_locator(mdates.MonthLocator([3, 6, 9, 12]))

    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)

plt.suptitle(f'STL Decomposition of {series_name}', fontsize=16)
plt.tight_layout()
plt.show()

# Detrend the series (remove trend component)
detrended_series = series - result.trend

# Seasonally difference the detrended series
seasonally_differenced_series = detrended_series.diff(periods=4).dropna()

# Plot the detrended and seasonally differenced series
plt.figure(figsize=(10, 5))
plt.plot(seasonally_differenced_series)
plt.title(f'Detrended and Seasonally Differenced Series of {series_name}')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the new series to a CSV file
output_file_name = f'{series_name}_detrended_and_seasonally_differenced.csv'
output_csv_path = os.path.join(csv_subdirectory, output_file_name)
seasonally_differenced_series.to_csv(output_csv_path, header=True)

# Function to perform ADF and KPSS tests
def test_stationarity(series):
    # ADF Test
    adf_result = adfuller(series)
    print(f"ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}")
    if adf_result[1] < 0.05:
        print("Series is stationary according to the ADF test (p < 0.05).")
    else:
        print("Series is not stationary according to the ADF test (p >= 0.05).")
    
    # KPSS Test
    kpss_result = kpss(series, regression='c')
    print(f"KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}")
    if kpss_result[1] < 0.05:
        print("Series is not stationary according to the KPSS test (p < 0.05).")
    else:
        print("Series is stationary according to the KPSS test (p >= 0.05).")

# Test stationarity for the detrended and seasonally differenced series
print(f"Stationarity test results for detrended and seasonally differenced series of {series_name}:")
test_stationarity(seasonally_differenced_series)