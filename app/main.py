import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config

# Set page config
st.set_page_config(
    page_title="Stock Market Trend Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url("https://images.unsplash.com/photo-1615992174118-9b8e9be025e7?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] {{
background-color: rgba(0, 0, 0, 0.8);
color: white;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
visibility: hidden;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function to set the background



# App title and description
st.title(" Stock Market Trend Predictor")
st.markdown("""
This app predicts future stock price trends using historical data and machine learning techniques.:

## How This App Works


---
###  Data Collection
- **Source**: Real-time stock data from Yahoo Finance API
- **Data Points**: Open, High, Low, Close prices and Volume
- **Coverage**: Any publicly traded stock (enter ticker symbol)
---
###  Technical Analysis
- **Moving Averages**: 5-day, 20-day, and 50-day for trend identification
- **RSI (Relative Strength Index)**: Detects overbought/oversold conditions
- **Volatility**: Calculated using rolling standard deviation of returns
---
###  Prediction Models
- **Random Forest Regression**: Advanced machine learning algorithm that combines multiple decision trees
- **Simple Moving Average**: Traditional statistical approach for trend projection
---
###  Trading Signals
- Buy/Sell recommendations based on technical indicator crossovers
- Overbought (RSI > 70) and oversold (RSI < 30) signals for potential reversals

---
**Get Started**: Select your stock, time period, and prediction model using the sidebar controls.
""")

with st.expander("📋 Example Stocks to Try"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Tech Stocks
        - **AAPL** - Apple Inc.
        - **MSFT** - Microsoft Corporation
        - **GOOGL** - Alphabet Inc. (Google)
        - **AMZN** - Amazon.com Inc.
        - **NVDA** - NVIDIA Corporation
        
        ### Financial Stocks
        - **JPM** - JPMorgan Chase & Co.
        - **BAC** - Bank of America Corp.
        - **V** - Visa Inc.
        - **MA** - Mastercard Inc.
        - **GS** - Goldman Sachs Group Inc.
        """)
        
    with col2:
        st.markdown("""
        ### Healthcare Stocks
        - **JNJ** - Johnson & Johnson
        - **PFE** - Pfizer Inc.
        - **MRNA** - Moderna Inc.
        - **UNH** - UnitedHealth Group Inc.
        - **CVS** - CVS Health Corp.
        
        ### Popular ETFs
        - **SPY** - SPDR S&P 500 ETF Trust
        - **QQQ** - Invesco QQQ (NASDAQ-100 Index)
        - **VTI** - Vanguard Total Stock Market ETF
        - **ARKK** - ARK Innovation ETF
        - **XLF** - Financial Select Sector SPDR Fund
        """)

# Sidebar for inputs
st.sidebar.header("Parameters")

# Stock symbol input
stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()

# Date range
today = datetime.now()
one_year_ago = today - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=one_year_ago)
end_date = st.sidebar.date_input("End Date", value=today)

# Model selection
model_type = st.sidebar.selectbox(
    "Prediction Model",
    ["Random Forest", "Simple Moving Average"]
)

# Prediction days
prediction_days = st.sidebar.slider("Prediction Days", 7, 60, 30)

# Function to load data
@st.cache_data
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        if data.empty:
            st.error(f"No data found for {symbol}. Please check the stock symbol.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None
# Helper functions
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def predict_with_random_forest(data, future_days):
    # Prepare features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'MA50', 'RSI', 'Return', 'Volatility']
    X = data[features]
    y = data['Close'].shift(-1)  # Predict next day's closing price
    X = X[:-1]  # Remove last row since we don't have the y for it
    y = y[:-1]  # Remove the last NaN value
    
    # Split data into training and testing sets (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on test data
    predictions = model.predict(X_test_scaled)
    
    # Calculate error metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    return predictions, data.iloc[train_size+1:train_size+len(predictions)+1], mse, mae

def predict_future(data, days):
    # Prepare features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'MA50', 'RSI', 'Return', 'Volatility']
    X = data[features]
    y = data['Close']
    
    # Scale the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    # Train the model on all available data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y.values.ravel())
    
    # Make predictions for future days
    future_predictions = []
    last_data = X.iloc[-1:].values
    current_prediction = last_data
    
    for _ in range(days):
        # Scale the input
        current_prediction_scaled = scaler_X.transform(current_prediction)
        
        # Make prediction
        next_pred = model.predict(current_prediction_scaled)[0]
        future_predictions.append(next_pred)
        
        # Update the prediction data for the next iteration (simple approach)
        # In a real system, we would build more sophisticated features
        current_prediction[0][0] = next_pred * 0.99  # Open (slightly lower than previous close)
        current_prediction[0][1] = next_pred * 1.01  # High (slightly higher than previous close)
        current_prediction[0][2] = next_pred * 0.98  # Low (lower than previous close)
        current_prediction[0][3] = next_pred  # Close
        # Keep other features like volume, MA, RSI etc. from the last known values
    
    return future_predictions

def calculate_prediction_confidence(predictions, historical_std):
    # Simple confidence calculation based on prediction volatility
    # Lower volatility in predictions relative to historical data suggests higher confidence
    prediction_std = np.std(predictions)
    
    # Return a confidence score between 0.3 and 0.9
    # Lower ratio of prediction_std to historical_std means higher confidence
    ratio = min(prediction_std / historical_std, 1)
    confidence = 0.9 - (ratio * 0.6)
    return max(0.3, confidence)

def predict_with_sma(data, days):
    # Calculate a moving average
    sma = data['Close'].rolling(window=20).mean().iloc[-1]
    
    # Calculate trend from last 20 days
    trend = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / 20
    
    # Project forward using the trend
    predictions = [sma + trend * i for i in range(1, days + 1)]
    
    # Calculate confidence based on how well SMA has predicted in the past
    last_20_days = data['Close'].iloc[-20:].values
    sma_20_days_ago = data['Close'].rolling(window=20).mean().iloc[-21]
    trend_20_days_ago = (data['Close'].iloc[-21] - data['Close'].iloc[-41]) / 20
    
    predicted_20_days = [sma_20_days_ago + trend_20_days_ago * i for i in range(1, 21)]
    accuracy = 1 - min(1, np.mean(np.abs((last_20_days - predicted_20_days) / last_20_days)))
    
    return predictions, accuracy

def generate_trading_signals(data):
    # Simple strategy: Buy when 5-day MA crosses above 20-day MA, Sell when it crosses below
    signals = pd.DataFrame(index=data.index)
    signals['Price'] = data['Close']
    signals['Signal'] = "Hold"
    
    # Buy signal: 5-day MA crosses above 20-day MA
    buy_signals = (data['MA5'] > data['MA20']) & (data['MA5'].shift(1) <= data['MA20'].shift(1))
    signals.loc[buy_signals, 'Signal'] = "Buy"
    
    # Sell signal: 5-day MA crosses below 20-day MA
    sell_signals = (data['MA5'] < data['MA20']) & (data['MA5'].shift(1) >= data['MA20'].shift(1))
    signals.loc[sell_signals, 'Signal'] = "Sell"
    
    # Add RSI signals
    # Oversold - additional buy signal
    signals.loc[data['RSI'] < 30, 'Signal'] = "Buy"
    
    # Overbought - additional sell signal
    signals.loc[data['RSI'] > 70, 'Signal'] = "Sell"
    
    return signals[['Price', 'Signal']].tail(10)
# Load data
if st.sidebar.button("Load Data & Predict"):
    # Show loading spinner
    with st.spinner('Loading data and preparing predictions...'):
        df = load_data(stock_symbol, start_date, end_date)
        
        if df is not None:
            # Display basic information
            st.subheader(f"{stock_symbol} Stock Data")
            
            # Company information
            try:
                stock_info = yf.Ticker(stock_symbol).info
                company_name = stock_info.get('longName', stock_symbol)
                st.markdown(f"### {company_name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${stock_info.get('currentPrice', 'N/A')}")
                with col2:
                    st.metric("52-Week High", f"${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
                with col3:
                    st.metric("52-Week Low", f"${stock_info.get('fiftyTwoWeekLow', 'N/A')}")
            except:
                st.write("Unable to fetch detailed company information")
            
            # Data overview
            st.subheader("Data Overview")
            st.dataframe(df.tail())
            
            # Create features for prediction
            df_processed = df.copy()
            df_processed['Date'] = df_processed.index
            
            # Add technical indicators
            df_processed['MA5'] = df_processed['Close'].rolling(window=5).mean()
            df_processed['MA20'] = df_processed['Close'].rolling(window=20).mean()
            df_processed['MA50'] = df_processed['Close'].rolling(window=50).mean()
            df_processed['RSI'] = calculate_rsi(df_processed['Close'], 14)
            df_processed['Return'] = df_processed['Close'].pct_change()
            df_processed['Volatility'] = df_processed['Return'].rolling(window=20).std()
            
            # Drop rows with NaN values
            df_processed = df_processed.dropna()
            
            # Interactive price chart
            st.subheader("Historical Price Chart")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, subplot_titles=('Price', 'Volume'),
                                row_heights=[0.7, 0.3])
            
            fig.add_trace(
                go.Candlestick(x=df.index,
                              open=df['Open'],
                              high=df['High'],
                              low=df['Low'],
                              close=df['Close'],
                              name="Candlestick"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df_processed.index, y=df_processed['MA20'], 
                           name="20-day MA", line=dict(color='orange')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df_processed.index, y=df_processed['MA50'], 
                           name="50-day MA", line=dict(color='green')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name="Volume"),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Make predictions based on selected model
            if model_type == "Random Forest":
                predictions, test_data, mse, mae = predict_with_random_forest(df_processed, prediction_days)
                
                st.subheader("Random Forest Prediction Results")
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"Mean Absolute Error: {mae:.4f}")
                
                # Plot predictions
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], 
                                         mode='lines', name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=test_data.index, y=predictions, 
                                         mode='lines', name='Predicted', line=dict(color='red')))
                fig.update_layout(title=f"{stock_symbol} Price Prediction (Test Period)",
                                 xaxis_title="Date",
                                 yaxis_title="Price",
                                 height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Future prediction
                future_predictions = predict_future(df_processed, prediction_days)
                
                # Create future dates
                last_date = df_processed.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
                future_df = pd.DataFrame(index=future_dates, data={'Predicted': future_predictions})
                
                # Plot future predictions
                fig = go.Figure()
                # Show last 60 days of actual data
                fig.add_trace(go.Scatter(x=df_processed.index[-60:], y=df_processed['Close'][-60:], 
                                         mode='lines', name='Historical', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted'], 
                                         mode='lines', name='Forecast', line=dict(color='red')))
                fig.update_layout(title=f"{stock_symbol} {prediction_days}-Day Price Forecast",
                                 xaxis_title="Date",
                                 yaxis_title="Price",
                                 height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Trend analysis
                predicted_trend = "Upward" if future_predictions[-1] > float(df_processed['Close'].iloc[-1]) else "Downward"
                trend_color = "green" if predicted_trend == "Upward" else "red"
                
                st.markdown(f"### Predicted Trend: <span style='color:{trend_color}'>{predicted_trend}</span>", unsafe_allow_html=True)
                
                # Prediction confidence
                confidence = calculate_prediction_confidence(future_predictions, float(df_processed['Close'].std()))
                st.progress(confidence)
                st.write(f"Prediction Confidence: {confidence:.1%}")
                
            else:  # Simple Moving Average
                predictions, confidence = predict_with_sma(df_processed, prediction_days)
                
                # Plot predictions
                fig = go.Figure()
                # Show last 60 days of actual data
                fig.add_trace(go.Scatter(x=df_processed.index[-60:], y=df_processed['Close'][-60:], 
                                         mode='lines', name='Historical', line=dict(color='blue')))
                
                # Show trend line for prediction period
                future_dates = pd.date_range(start=df_processed.index[-1] + pd.Timedelta(days=1), periods=prediction_days)
                fig.add_trace(go.Scatter(x=future_dates, y=predictions, 
                                         mode='lines', name='SMA Forecast', line=dict(color='red')))
                
                fig.update_layout(title=f"{stock_symbol} {prediction_days}-Day SMA Forecast",
                                 xaxis_title="Date",
                                 yaxis_title="Price",
                                 height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Trend analysis
                predicted_trend = "Upward" if predictions[-1] > float(df_processed['Close'].iloc[-1]) else "Downward"
                trend_color = "green" if predicted_trend == "Upward" else "red"
                
                st.markdown(f"### Predicted Trend: <span style='color:{trend_color}'>{predicted_trend}</span>", unsafe_allow_html=True)
                st.progress(confidence)
                st.write(f"Prediction Confidence: {confidence:.1%}")
            
            # Display trading signals
            st.subheader("Trading Signals")
            signals = generate_trading_signals(df_processed)
            
            # Color code the signals
            def color_signal(val):
                if val == "Buy":
                    return 'background-color: green; color: white'
                elif val == "Sell":
                    return 'background-color: red; color: white'
                elif val == "Hold":
                    return 'background-color: yellow; color: black'
                return ''
            
            st.dataframe(signals.style.applymap(color_signal, subset=['Signal']))
            
            # Disclaimer
            st.info("⚠️ Disclaimer: This tool is for demonstration purposes only. Do not use it for real investment decisions.")




# Cache clearing button in sidebar
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")

# Add information about the app
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This app demonstrates a simple stock market trend predictor using machine learning.
- Data source: Yahoo Finance
- Models: Random Forest, Simple Moving Average
- Features: Price data, technical indicators
""")

# Show a message when no data is loaded
if 'button' not in st.session_state:
    st.info("👈 Enter a stock symbol and click 'Load Data & Predict' to get started.")