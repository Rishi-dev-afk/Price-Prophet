import torch
import numpy as np
from model import StockPredictor
from data_preprocessing import load_and_clean_data
from torch.utils.data import DataLoader, TensorDataset

# Parameters
stock_symbol = "AAPL"  # Example: Apple stock
start_date = "2022-01-01"
end_date = "2023-01-01"
epochs = 50
batch_size = 32
learning_rate = 0.001

# Load and preprocess data
data, scaler = load_and_clean_data(stock_symbol, start_date, end_date)
data = torch.tensor(data.values, dtype=torch.float32)

# Create sequences for LSTM
def create_sequences(data, seq_length=30):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return torch.stack(x), torch.stack(y)

seq_length = 30
x, y = create_sequences(data, seq_length)
train_data = TensorDataset(x, y)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = StockPredictor().to(torch.device("cpu"))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), "../models/stock_predictor.pth")
