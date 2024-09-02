import pandas as pd
from xbbg import blp
from datetime import datetime, timedelta

def check_forward_tickers():
    # Define the tickers for forward 10Y rates
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

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Smaller range for testing

    for ticker in fwd_6M_IRS.values():
        try:
            data = blp.bdh(tickers=ticker, flds='PX_LAST', start_date=start_date, end_date=end_date)
            if data.empty:
                print(f"No data fetched for ticker: {ticker}")
            else:
                print(f"Data fetched successfully for ticker: {ticker}")
                print(data.head())
        except Exception as e:
            print(f"Error fetching data for ticker {ticker}: {e}")

check_forward_tickers()