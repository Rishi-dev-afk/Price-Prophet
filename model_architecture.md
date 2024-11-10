The core of the Price Prophet stock prediction model is an LSTM-based architecture designed to handle sequential time-series data.

LSTM Layers: The model includes 4 LSTM layers, which are suitable for capturing long-term dependencies in the stock price time series. 

LSTMs are effective in understanding sequential data due to their ability to retain information over time.

Hidden Units: Each LSTM layer contains 128 hidden units, ensuring sufficient capacity to model complex patterns in the stock price data.

Dropout: A dropout layer is added after each LSTM layer (with a rate of 0.3) to prevent overfitting by randomly deactivating certain neurons during training.

Batch Normalization: Applied to the output of the last LSTM layer, this helps to stabilize the training process by normalizing the output activations.

Fully Connected Layer: A linear layer follows the LSTM output, which produces the final stock price prediction for each stock symbol. 

The number of outputs matches the number of stock symbols in the dataset.

This architecture is tailored for the task of stock price prediction, leveraging LSTMs' ability to learn temporal patterns while minimizing the risk 
of overfitting through regularization techniques like dropout and batch normalization.
