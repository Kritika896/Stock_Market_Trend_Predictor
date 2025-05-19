# Stock Market Trend Predictor

*A Streamlit web application for predicting future stock price trends using historical data and machine learning.*

---

## 📊 Overview
This application provides an interactive dashboard to analyze historical stock data and predict future price movements using machine learning models. It's designed as a demonstration tool for **educational purposes**, showcasing how **time series prediction** can be applied to financial markets.

![image alt](https://github.com/Kritika896/Stock_Market_Trend_Predictor/blob/main/1745671452162.jpeg?raw=true)
---

## 🚀 Features

- **Real-time Stock Data**  
  Fetch historical stock data for any publicly traded company using the Yahoo Finance API.

- **Technical Indicators**  
  Automatically calculates and displays key technical indicators:
  - Moving Averages (5-day, 20-day, 50-day)
  - Relative Strength Index (RSI)

- **Interactive Visualizations**  
  Display stock price history with moving averages.

- **Dual Prediction Models**
  - Random Forest Regressor *(machine learning approach)*
  - Simple Moving Average *(traditional technical analysis approach)*

- **Model Validation**  
  Tests model accuracy on historical data.

- **Future Price Forecasting**  
  Predicts stock prices for the next 30 days.

- **Trend Analysis**  
  Identifies potential upward or downward price trends.

- **Performance Metrics**  
  Displays prediction accuracy metrics.

---

## 🛠️ Installation

### Prerequisites
- Python 3.7+  
- `pip` (Python package installer)

## 💡 Usage

1. Enter a valid stock symbol (e.g., `AAPL` for Apple, `MSFT` for Microsoft) in the sidebar.
2. Click **"Load Data & Predict"** to fetch and analyze the stock data.
3. Review the historical data, technical indicators, and price charts.
4. Examine the price predictions and trend analysis from both models.
5. Use the **"Clear Cache"** button to refresh and analyze a different stock.

---

## 🔬 How It Works

### 📈 Data Collection
- Uses **Yahoo Finance API** via the `yfinance` library to fetch 1 year of historical data.

### 📉 Technical Analysis
Calculates several commonly used technical indicators:

- **Moving Averages**  
  Identifies price trends over 5, 20, and 50-day windows.

- **RSI (Relative Strength Index)**  
  Measures momentum to identify overbought or oversold conditions.

---

### 🤖 Prediction Models

#### ✅ Random Forest Model
- Uses features like Open, High, Low, Close, Volume, Moving Averages, RSI.
- Trains on historical data to identify patterns.
- Predicts future prices based on learned relationships.

#### 📊 Simple Moving Average Model
- Projects trends using recent price averages.
- Simpler but sometimes equally effective for short-term predictions.

---

### ✔️ Validation
- Splits historical data into training and testing sets to assess model accuracy.

---

## ⚠️ Limitations

- Predictions are based solely on historical price data.
- Cannot account for unexpected news or events.
- Does not include fundamental or sentiment analysis.
- **Educational use only** – not intended for actual trading or investment decisions.

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit a Pull Request:

1. **Fork the repository**

2. **Create your feature branch**
   ```bash
   git checkout -b feature/amazing-feature

