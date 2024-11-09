import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

def load_and_clean_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data[['Close']]
    data.dropna(inplace=True)
    return data

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

def create_sequences(data, seq_length=60):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(torch.tensor(data[i:i+seq_length], dtype=torch.float32))
        y.append(torch.tensor(data[i+seq_length], dtype=torch.float32))
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y
  
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, output_size=1, dropout=0.3):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

data = load_and_clean_data("AAPL", "2008-01-01", "2023-01-01")
normalized_data, scaler = normalize_data(data)
x, y = create_sequences(normalized_data, seq_length=60)

train_size = int(len(x) * 0.8)
train_x, test_x = x[:train_size], x[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = StockLSTM(input_size=1, hidden_size=64, num_layers=3, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=30):
    for epoch in range(epochs):
        model.train()  
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)          
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        evaluate_model(model, test_loader, criterion)

def evaluate_model(model, test_loader, criterion):
    model.eval() 
    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

def get_predictions(model, test_loader, scaler):
    model.eval() 
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))
    return predictions, actuals

def plot_predictions(predictions, actuals):
    plt.figure(figsize=(14, 7))
    plt.plot(actuals, label="Real Price", color="b")
    plt.plot(predictions, label="Predicted Price", color="r")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.title("Predicted vs Real Stock Prices")
    plt.show()

train_model(model, train_loader, test_loader, criterion, optimizer, epochs=30)
predictions, actuals = get_predictions(model, test_loader, scaler)
plot_predictions(predictions, actuals)

def predict(model, input_sequence, scaler):
    model.eval()  
    with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0).to(device)  
        prediction = model(input_sequence)
        prediction = scaler.inverse_transform(prediction.cpu().numpy())  
        return prediction[0][0]

last_sequence = test_x[-1]  
predicted_price = predict(model, last_sequence, scaler)
print(f"Predicted next price: {predicted_price:.2f}")

