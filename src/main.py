import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create DataLoaders for training and testing
batch_size = 32
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = StockLSTM(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

        # Evaluate on test data after each epoch
        evaluate_model(model, test_loader, criterion)

# Evaluation function
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

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=50)

# Prediction function for a new sequence (e.g., last known sequence in test data)
def predict(model, input_sequence, scaler):
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0).to(device)
        prediction = model(input_sequence)
        prediction = scaler.inverse_transform(prediction.cpu().numpy())  # Rescale prediction
        return prediction[0][0]

# Example usage for prediction
last_sequence = test_x[-1]  # Example last sequence in the test set
predicted_price = predict(model, last_sequence, scaler)
print(f"Predicted next price: {predicted_price:.2f}")
