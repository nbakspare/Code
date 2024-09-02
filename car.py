import pandas as pd
from xbbg import blp
from datetime import datetime, timedelta

def fetch_historical_data():
    # Define the tickers for spot and forward 10Y rates
    spot_IRS = {
        'USD': 'USSW10 Curncy',
        'EUR': 'EUSA10 Curncy',
        'JPY': 'JYSW10 Curncy',
        'GBP': 'BPSW10 Curncy',
        'AUD': 'ADSW10Q Curncy',
        'CAD': 'CDSW10 Curncy',
        'SEK': 'SKSW10 Curncy',
        'CHF': 'SFSW10  Curncy',
        'NOK': 'NKSW10 Curncy',
        'NZD': 'NDSWAP10 Curncy',
    }

    fwd_6M_IRS = {
        'USD': 'USFS0F10 Curncy',
        'EUR': 'EUSAF10 Curncy',
        'JPY': 'JYFS0F10 Curncy',
        'GBP': 'BPFS0F10 Curncy',
        'AUD': 'ADFS0F10 Curncy',
        'CAD': 'CDFS0F10 Curncy',
        'SEK': 'SKFS0F10 Curncy',
        'CHF': 'SFFS0F10 Curncy',
        'NOK': 'NKFS0F10 Curncy',
        'NZD': 'NDFS0F10 Curncy',
    }

    # Define a smaller date range for testing
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Smaller range for testing

    # Fetch historical swap rates and forward rates
    try:
        swap_rates = blp.bdh(tickers=list(spot_IRS.values()), flds='PX_LAST', start_date=start_date, end_date=end_date)
        swap_6M_fwd_rates = blp.bdh(tickers=list(fwd_6M_IRS.values()), flds='PX_LAST', start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Debugging: Print fetched data
    print("Swap Rates Data:")
    print(swap_rates.head())
    print("6M Forward Swap Rates Data:")
    print(swap_6M_fwd_rates.head())

    # Check if the forward rates DataFrame is empty
    if swap_6M_fwd_rates.empty:
        print("6M Forward Swap Rates Data is empty. Please check the tickers or data fetch issue.")
        print("Attempted Forward Rate Tickers:")
        print(list(fwd_6M_IRS.values()))
        return

    # Rename columns for spot rates
    swap_rates.columns = [spot_IRS.get(ticker.split()[0], ticker) for ticker in swap_rates.columns.get_level_values(0)]

    # Rename columns for forward rates
    swap_6M_fwd_rates.columns = [spot_IRS.get(list(fwd_6M_IRS.keys())[list(fwd_6M_IRS.values()).index(ticker)], ticker) for ticker in swap_6M_fwd_rates.columns.get_level_values(0)]

    # Debugging: Print renamed columns
    print("Renamed Swap Rates Columns:")
    print(swap_rates.columns)
    print("Renamed 6M Forward Swap Rates Columns:")
    print(swap_6M_fwd_rates.columns)

    # Fetch durations
    durations = blp.bdp(tickers=list(spot_IRS.values()), flds=['DUR_ADJ_BID'])
    print("Durations Data:")
    print(durations)

    # Convert columns to lowercase and check for 'dur_adj_bid'
    durations.columns = durations.columns.str.lower()
    if 'dur_adj_bid' not in durations.columns:
        raise ValueError("The 'dur_adj_bid' field is not available in the returned data.")

    durations.index = [spot_IRS.get(ticker, ticker) for ticker in durations.index]
    durations = durations['dur_adj_bid']

    # Debugging: Print durations data
    print("Processed Durations Data:")
    print(durations)

    # Calculate carry
    carry = pd.DataFrame()
    for ticker in swap_rates.columns:
        if ticker in swap_6M_fwd_rates.columns and ticker in durations.index:
            print(f"Calculating carry for ticker: {ticker}")
            carry[ticker] = durations[ticker] * (swap_6M_fwd_rates[ticker] - swap_rates[ticker]) / 100

    carry = carry.dropna(how='all')  # Drop rows where all values are NaN

    # Debugging: Print carry data
    print("Calculated Carry Data:")
    print(carry.head())

    # Save carry data to CSV
    carry.to_csv('swap_6M_carry_last_30_days.csv')

    print("Historical carry data for the last 30 days has been downloaded and saved to CSV files.")

# Run the function to fetch and save historical data
fetch_historical_data()
