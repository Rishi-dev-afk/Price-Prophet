This project utilizes deep learning techniques, specifically Long Short-Term Memory (LSTM) networks, to predict stock prices for multiple companies based on historical data. Stock price prediction is a critical task in the financial sector, and LSTM models are well-suited for such time-series forecasting due to their ability to retain and learn from long-term dependencies in sequential data.

Key Components:
Data Collection and Preprocessing:

The project pulls historical stock data for multiple companies (such as AAPL, MSFT, GOOGL) using the Yahoo Finance API (yfinance).
The data consists of daily closing prices, which are then cleaned and normalized using MinMaxScaler to ensure that all features lie within the same scale. Missing values are dropped to ensure the model is trained on valid data.
Model Architecture:

The core of this project is an LSTM-based neural network (StockLSTM class). The model consists of multiple LSTM layers followed by a fully connected layer, which outputs the predicted closing prices.
The architecture is designed to handle multiple input stocks (e.g., AAPL, MSFT, GOOGL) simultaneously and predict their future prices.
A dropout layer and batch normalization are included to avoid overfitting and improve generalization.
Training:

The model is trained on 80% of the data and tested on the remaining 20%. The training process includes gradient descent optimization using the Adam optimizer, and the loss is minimized using Mean Squared Error (MSE) loss.
The model is evaluated after each epoch, and the best-performing model (lowest loss) is saved for later use.
Prediction:

Once trained, the model predicts the next day's stock prices for the given symbols. The predictions are scaled back to their original values using the scaler.
The project includes a function to visualize the real vs predicted stock prices, providing insights into the model's performance over time.
Streamlit Interface:

The project is wrapped into a user-friendly web application using Streamlit, allowing users to interact with the model easily.
Users can input stock symbols, select a date range, and adjust the sequence length for the LSTM model through the app interface.
The predicted stock prices for the next day are displayed, and users can also view the plotted graph comparing real and predicted stock prices.
Conclusion:
This project demonstrates the use of LSTM networks for stock price prediction, leveraging historical data to forecast future trends. The integration with Streamlit makes the model accessible and interactive for users. While the model works with stock data from multiple companies, it can be extended to include more sophisticated features, such as technical indicators or news sentiment analysis, to improve prediction accuracy.
