import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define the file path
file_path = 'CSV Data/Seasonally_Differenced_Data.csv'

# Load the CSV data into a DataFrame
df = pd.read_csv(file_path)

# Identify the columns to be standardized (excluding the date column and header)
columns_to_standardize = df.columns[1:]  # Assuming the first column is the date

# Initialize the scaler
scaler = StandardScaler()

# Standardize only the specified columns
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

# Overwrite the original file with the standardized data
df.to_csv(file_path, index=False)

print("Data has been standardized (excluding header and date column) and saved to the original file.")
