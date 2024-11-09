import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data[['Close']]  # Only keep closing prices
    data.dropna(inplace=True)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close'] = scaler.fit_transform(data[['Close']])
    
    return data, scaler

