import yfinance as yf
import pandas as pd
import sys
import sqlite3
# Define the ticker symbols for the indices and Gold
tickers = {
    "S&P 500": "^GSPC",
    "MSCI ACWI": "ACWI",
    "MSCI EM": "EEM",
    "Gold": "GC=F"  # Added ticker for Gold
}

# Function to download and resample closing price data to monthly frequency
def download_monthly_closing_prices(tickers, start_date, end_date):
    data = {}
    for name, ticker in tickers.items():
        print(f"Downloading data for {name} ({ticker})...")
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                # Resample to monthly frequency and select the last closing price of each month
                monthly_data = stock_data['Close'].resample('M').last()
                data[name] = monthly_data
            else:
                print(f"Warning: No data found for {name} ({ticker}).")
        except Exception as e:
            print(f"Error downloading data for {name} ({ticker}): {e}")

    if not data:
        raise ValueError("No valid data was downloaded. Please check the ticker symbols or date range.")

    # Combine all monthly data into a single DataFrame
    combined_data = pd.concat(data, axis=1)  # Combine Series into a DataFrame
    return combined_data

# Define date range
start_date = "2010-01-01"
end_date = "2024-11-30"

# Download data
try:
    closing_prices_monthly = download_monthly_closing_prices(tickers, start_date, end_date)
    # Display the first 10 rows
    print("First 10 rows of the monthly closing prices data:")
    print(closing_prices_monthly.head(10))
    # Save to a CSV file
    closing_prices_monthly.to_csv("monthly_closing_prices_with_gold.csv")
    print("Data downloaded and saved to monthly_closing_prices_with_gold.csv")
except ValueError as e:
    print(e)

############################################################################################################
# Cleaning Equity index price data 
eqtyData = closing_prices_monthly.iloc[:, :3]
eqtyData.columns = ['S&P500', 'ACWI', 'MSCIEM']
print(eqtyData)



# Calculating monthly price returns 
eqtyDataMthly = eqtyData.pct_change().dropna()
eqtyDataMthly.sort_index(ascending=False, inplace=True)
print(eqtyDataMthly)


############################################################################################################
# Loading csv from the macro variables file
# Define the folder path and file name
#folder_path = "C:/Users/Arif/Documents/portfolio/portfolio_qd"  # Replace with your folder path
file_name = "macro.csv"          # Replace with your CSV file name
file_path = f"{file_name}"
import pandas as pd




#############################################################################################################
# Build a relational database in any publicly available DB
# Load equity monthly returns data into the DB
# Load macro data into DB
sqlite_db = "financial_data.db"
table_name = "equity_indices"

def table_exists(connection, table_name):
    """
    Check if a table exists in the SQLite database.
    :param connection: SQLite connection object.
    :param table_name: Name of the table to check.
    :return: True if the table exists, False otherwise.
    """
    query = """
    SELECT name FROM sqlite_master WHERE type='table' AND name=?;
    """
    cursor = connection.cursor()
    cursor.execute(query, (table_name,))
    result = cursor.fetchone()
    return result is not None

def storeEquitData():
    # Add index as a column for storage
    eqtyData.reset_index(inplace=True)
    eqtyData.rename(columns={"index": "date"}, inplace=True)

    # SQLite database file name
    

    # Connect to SQLite database (creates the database file if it doesn't exist)
    connection = sqlite3.connect(sqlite_db)

    # Store eqtyData in the SQLite database
    
    try:
        if table_exists(connection, table_name):
            print(f"Table '{table_name}' already exists. Skipping data insertion.")
        else:
            # Write DataFrame to SQLite
            eqtyData.to_sql(table_name, connection, if_exists="replace", index=False)
            print(f"Data successfully stored in table '{table_name}' of {sqlite_db}")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        connection.close()

def retreiveDBData(table_name):
    # To verify data storage
    # Reconnect and read data back from the SQLite database
    connection = sqlite3.connect(sqlite_db)
    retrieved_data = pd.read_sql(f"SELECT * FROM {table_name}", connection)
    connection.close()

    # Display the retrieved data
    print("\nRetrieved Data for ", table_name, " : ")
    print(retrieved_data)

storeEquitData()
retreiveDBData(table_name)

# File name for the macro CSV
file_name = "macro.csv"
table_name= "macros"
try:
    # Load the CSV file into a pandas DataFrame
    macro_data = pd.read_csv(file_name)
    
    # Convert 'Date' column to datetime if it exists
    if 'Date' in macro_data.columns:
        macro_data['Date'] = pd.to_datetime(macro_data['Date'])
    else:
        raise KeyError("The 'Date' column is missing from the CSV file.")
    
    # Ensure the DataFrame has no missing values
    macro_data.dropna(inplace=True)
    
    # Connect to SQLite database
    connection = sqlite3.connect(sqlite_db)
    
    # Check if the table already exists in the SQLite database
    def table_exists(conn, table_name):
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
        cursor = conn.cursor()
        cursor.execute(query, (table_name,))
        return cursor.fetchone() is not None
    
    # Insert data into the SQLite database
    if table_exists(connection, table_name):
        print(f"Table '{table_name}' already exists. Skipping data insertion.")
    else:
        macro_data.to_sql(table_name, connection, if_exists="replace", index=False)
        print(f"Data successfully stored in table '{table_name}' of {sqlite_db}.")
    
    # Close the connection
    connection.close()

except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
except KeyError as e:
    print(f"Error: {e}")
except pd.errors.EmptyDataError:
    print(f"Error: File '{file_name}' is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

retreiveDBData(table_name)
print('done')
sys.exit(1)




#############################################################################################################
# Creating portfolio returns function 

# Function to calculate portfolio returns
def calculate_portfolio_returns(asset_returns, weight1, weight2, weight3):
    """
    Calculate portfolio returns based on user-defined weights for three assets.

    Parameters:
        asset_returns (pd.DataFrame): A DataFrame containing the time series of returns for 3 assets with columns ['Asset 1', 'Asset 2', 'Asset 3'].
        weight1 (float): Weight of Asset 1 in the portfolio (percentage).
        weight2 (float): Weight of Asset 2 in the portfolio (percentage).
        weight3 (float): Weight of Asset 3 in the portfolio (percentage).

    Returns:
        pd.Series: A time series of portfolio returns.
    """
    # Check if weights sum to 100%
    total_weight = weight1 + weight2 + weight3
    if total_weight != 100:
        raise ValueError(f"The weights must sum to 100%. Current sum: {total_weight}%")

    # Convert weights to decimal form
    weight1 /= 100
    weight2 /= 100
    weight3 /= 100

    # Calculate portfolio returns
    portfolio_returns = (
        weight1 * asset_returns['S&P500'] +
        weight2 * asset_returns['ACWI'] +
        weight3 * asset_returns['MSCIEM']
    )

    return portfolio_returns


try:
   # eqtyDataMthly['PtfRtns'] = calculate_portfolio_returns(eqtyDataMthly, 40, 30, 30)
    portfolio_Rtn = pd.DataFrame(calculate_portfolio_returns(eqtyDataMthly, 40, 30, 30), columns=['PtfRtns'])
    portfolio_Rtn.index = portfolio_Rtn.index.strftime('%Y-%m-%d')
    print("\nPortfolio Returns:")
    print(portfolio_Rtn)
except ValueError as e:
    print(f"Error: {e}")



#############################################################################################################################################################
# Doing lasso regression for portfolio returns against macro variables 
# Independent variables are macro variables, dependent variable is portfolio return 
#

# Creating the retruns and macri variaables dataframe
df = pd.concat([portfolio_Rtn, macro], axis=1)
df = df.dropna()



import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Example DataFrame (replace this with your actual data)
dates = pd.date_range(start="2000-01-01", periods=100, freq="M")
data = {
    'Portfolio_Returns': np.random.normal(0, 0.01, len(dates)),
    'Macro_Variable_1': np.random.normal(0, 0.02, len(dates)),
    'Macro_Variable_2': np.random.normal(0, 0.02, len(dates)),
    'Macro_Variable_3': np.random.normal(0, 0.02, len(dates))
}
df = pd.DataFrame(data, index=dates)

# Rolling window size
rolling_window = 24

# Initialize results storage
rolling_betas = []
rolling_correlations = []

# Standardize the data
def standardize(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

# Perform rolling Lasso regression
for start in range(len(df) - rolling_window + 1):
    # Select rolling window data
    df_window = df.iloc[start:start + rolling_window]

    # Define dependent and independent variables
    y = df_window['Portfolio_Returns']
    X = df_window.drop(columns=['Portfolio_Returns'])

    # Standardize independent variables
    X_scaled = standardize(X)

    # Fit Lasso regression
    lasso = Lasso(alpha=0.1, random_state=42)  # Adjust alpha as needed
    lasso.fit(X_scaled, y)

    # Store regression coefficients (betas)
    betas = pd.Series(lasso.coef_, index=X.columns, name=df.index[start + rolling_window - 1])
    rolling_betas.append(betas)

    # Store rolling correlations between variables
    correlations = X.corr()
    rolling_correlations.append(correlations)

# Combine rolling betas into a DataFrame
rolling_betas_df = pd.DataFrame(rolling_betas)

# Plot rolling betas
rolling_betas_df.plot(figsize=(12, 6), title="Rolling Betas")
plt.xlabel("Date")
plt.ylabel("Beta Coefficient")
plt.grid()
plt.show()

# Plot rolling correlations
for i, corr_matrix in enumerate(rolling_correlations):
    if i % 10 == 0:  # Plot every 10th window for visualization
        plt.figure(figsize=(6, 5))
        plt.title(f"Rolling Correlations (Window Ending {df.index[rolling_window + i - 1]})")
        plt.imshow(corr_matrix, cmap="coolwarm", interpolation="none")
        plt.colorbar(label="Correlation")
        plt.xticks(ticks=range(len(corr_matrix)), labels=corr_matrix.columns, rotation=45)
        plt.yticks(ticks=range(len(corr_matrix)), labels=corr_matrix.columns)
        plt.show()

# Summary and Explanation of Results
print("Summary of Rolling Regression Results:")
for i, betas in rolling_betas_df.iterrows():
    print(f"Window Ending {i}: Relevant Variables:")
    significant_vars = betas[betas != 0].index.tolist()
    print(f"  Variables with Non-Zero Coefficients: {significant_vars}")
    print(f"  Coefficients: {betas[betas != 0].to_dict()}")

# Explanation:
print("\nExplanation:")
print("The rolling regression identifies how the portfolio returns are influenced by macro variables in different time windows.")
print("\n- For each window, the Lasso regression performs variable selection by shrinking some coefficients to zero.")
print("\n- The rolling betas plot shows the time-varying contribution of each macro variable to portfolio returns.")
print("\n- Rolling correlations reveal interdependencies among macro variables, helping to understand how they move together.")
