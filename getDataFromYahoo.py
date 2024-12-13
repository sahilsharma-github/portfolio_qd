import yfinance as yf
import pandas as pd

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
end_date = "2024-12-31"

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
