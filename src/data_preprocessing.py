import torch
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
import sklearn
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data[['Close']] 
    data.dropna(inplace=True)  
    return data

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))  
    normalized_data = scaler.fit_transform(data)  
    return normalized_data, scaler

def create_sequences(data, seq_length=30):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(torch.tensor(data[i:i+seq_length], dtype=torch.float32))  
        y.append(torch.tensor(data[i+seq_length], dtype=torch.float32)) 
    return torch.stack(x), torch.stack(y)  

# Example usage
data = load_and_clean_data("AAPL", "2022-01-01", "2023-01-01")  
normalized_data, scaler = normalize_data(data)  
x, y = create_sequences(normalized_data)

train_size = int(len(x) * 0.8)
train_x, test_x = x[:train_size], x[train_size:]
train_y, test_y = y[:train_size], y[train_size:]


