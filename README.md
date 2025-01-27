# Stock Price Prediction with Machine Learning

## Overview
This project leverages machine learning and deep learning to predict stock prices based on historical data. It integrates a user-friendly web interface built with Streamlit, allowing users to analyze stock market trends and view predictions. The model has been trained on historical stock prices and demonstrates the use of LSTM (Long Short-Term Memory) neural networks for time series prediction.

## Features
- **Stock Data Visualization:** Displays historical stock prices with moving averages (MA50, MA100, MA200).
- **Interactive Input:** Users can input any stock symbol (e.g., GOOG) to fetch data.
- **Machine Learning Model:** Predicts future stock prices using an LSTM-based deep learning model.
- **Streamlit Interface:** A simple and intuitive interface for real-time interaction.

## Tech Stack
- **Frontend & Deployment:** Streamlit
- **Backend:** TensorFlow, Keras, Python
- **Data Source:** Yahoo Finance (via `yfinance` library)
- **Libraries Used:**
  - Numpy
  - Pandas
  - Matplotlib
  - Scikit-learn
  - TensorFlow/Keras

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/RhydhamPatel/Stock-Price-Prediction.git
   cd stock-price-prediction
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Enter the stock symbol (e.g., `GOOG`) in the input field.
2. The app fetches historical stock data from Yahoo Finance.
3. Visualize stock data with moving averages (MA50, MA100, MA200).
4. View predicted stock prices compared with actual prices.

## Model
The deep learning model used in this project is a pre-trained LSTM network. The model takes scaled input sequences of stock prices and outputs the predicted price.

- **Training Data:** The data is split into training (80%) and testing (20%) sets.
- **Scaling:** MinMaxScaler is used to scale data between 0 and 1 for better model performance.
- **Prediction:** The model predicts stock prices using the past 100 days of data.

## Visualizations
- **Price vs MA50:** Compare actual prices with a 50-day moving average.
- **Price vs MA50 vs MA100:** Include a 100-day moving average for a broader view.
- **Price vs MA100 vs MA200:** Add a 200-day moving average for long-term trends.
- **Original Price vs Predicted Price:** View how well the model predicts prices.

## Folder Structure
```
Stock-Price-Prediction/
|-- app.py                     # Streamlit app
|-- Stock Predictions Model.keras  # Pre-trained LSTM model
|-- stock-price.ipynb          # Jupyter Notebook for model training and experimentation
|-- requirements.txt           # List of dependencies
```

## Screenshots
### 1. Price vs MA-50
![Price vs MA50](https://github.com/user-attachments/assets/16111c51-4fa1-41b9-a79e-7a97f6e8d8d2)

### 2. Moving Average 50-100
![MA-50-100](https://github.com/user-attachments/assets/0fef6f65-355d-4226-943e-0e04c75e3e41)

### 3. Price Prediction
![Price Prediction](https://github.com/user-attachments/assets/167277d3-e5e3-4508-9aa2-05e3a2093876)

## Future Improvements
- **Incorporate Real-Time Data:** Integrate live stock market data.
- **Add More Indicators:** Include RSI, MACD, and Bollinger Bands.
- **Model Enhancements:** Experiment with advanced architectures and hyperparameter tuning.
- **Performance Metrics:** Display metrics such as RMSE, MAE, and R-squared.

## Author
[Rhydham](https://github.com/RhydhamPatel)

## License
This project is licensed under the MIT License.

## Acknowledgments
- Data sourced from [Yahoo Finance](https://finance.yahoo.com/).
- Inspired by the versatility of deep learning in time series forecasting.

---
Feel free to contribute by opening issues or submitting pull requests!

