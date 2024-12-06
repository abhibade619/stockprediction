import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Add custom styling using Streamlit's HTML and CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    h1, h2, h3 {
        color: #003366;
        text-align: center;
    }
    .stButton > button {
        background-color: #003366;
        color: white;
        border-radius: 10px;
    }
    .stDataFrame {
        border: 2px solid #003366;
        border-radius: 5px;
    }
    .stPlotlyChart {
        border: 2px solid #003366;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Header
st.title('🚀 Stock Price Prediction')
st.markdown(
    "#### Predict historical and future stock prices with machine learning! Analyze trends using **Moving Averages** and visualize predictions interactively."
)

# Sidebar Inputs
st.sidebar.header('Stock Selection')
ticker = st.sidebar.text_input("Enter Stock Ticker", value="NKE")  # Default is Nike
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2000-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-09-30"))

# Load Model
model = load_model('Nike_Price_Prediction_Model.keras')

# Fetch data based on user input
st.sidebar.write(f"Fetching data for ticker: `{ticker}`")
data = pd.DataFrame(yf.download(ticker, start=start_date, end=end_date))
data.reset_index(inplace=True)
data['Date'] = data['Date'].dt.date  # Remove time from the Date column

# Display historical data
st.subheader(f'📈 Historical Data for {ticker}')
st.write("Here is the raw historical stock price data.")
st.dataframe(data.style.format({'Close': '{:.2f}'}), height=300)

# Add Moving Averages
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_20'] = data['Close'].rolling(window=20).mean()
data.bfill(inplace=True)

# Combined Line Chart: Close, MA_10, MA_20
st.subheader('📊 Combined Moving Averages')
combined_chart = data[['Date', 'Close', 'MA_10', 'MA_20']]
combined_chart.set_index('Date', inplace=True)
combined_chart.columns = ['Close', 'MA_10', 'MA_20']
st.line_chart(combined_chart)

# Preprocess Data
train_data = data[:-200]  # Train on all but the last 200 rows
test_data = data[-200:]  # Test on the last 200 rows

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data[['Close', 'MA_10', 'MA_20']])
test_data_scaled = scaler.transform(test_data[['Close', 'MA_10', 'MA_20']])

# Prepare Test Data
base_days = 100
x_test, y_test = [], []

for i in range(base_days, test_data_scaled.shape[0]):
    x_test.append(test_data_scaled[i-base_days:i])
    y_test.append(test_data_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 3))  # 3 features

# Predict on Test Data
st.subheader(f'📉 Actual vs Predicted Prices for {ticker}')
st.write("Here are the predictions compared to the actual prices:")
pred_test = model.predict(x_test)

# Inverse transform predictions and actual values
pred_test = scaler.inverse_transform(
    np.concatenate([pred_test, np.zeros((pred_test.shape[0], 2))], axis=1)
)[:, 0]
y_actual = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 2))], axis=1)
)[:, 0]

# Prepare the prediction DataFrame with proper date formatting
test_dates = test_data.iloc[base_days:]['Date'].values  # Get corresponding test dates
pred_df = pd.DataFrame({'Date': test_dates, 'Actual': y_actual, 'Predicted': pred_test})
pred_df['Date'] = pd.to_datetime(pred_df['Date']).dt.date  # Remove time from Date

# Display the table with formatted dates
st.dataframe(pred_df.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}'}))

# Plot the chart with dates on the x-axis
st.markdown("**Actual vs Predicted Prices**")
st.line_chart(pred_df.set_index('Date'))

# Future Price Prediction
st.subheader(f'🔮 Future Stock Price Prediction for {ticker}')
st.write("Predicted prices for the next 30 business days.")
future_prices = []
m = test_data_scaled[-base_days:]  # Start with the last base_days of test data

for _ in range(30):
    inter = np.array([m[-base_days:]])
    pred = model.predict(inter)
    future_prices.append(pred[0][0])
    new_row = np.array([pred[0][0], m[-1, 1], m[-1, 2]])  # Keep MA_10 and MA_20 static
    m = np.vstack((m, new_row))

future_prices = scaler.inverse_transform(
    np.concatenate([np.array(future_prices).reshape(-1, 1), np.zeros((30, 2))], axis=1)
)[:, 0]

# Generate Future Dates
last_date = data['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=31, freq='B')[1:]
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices})
future_df['Date'] = future_df['Date'].dt.date  # Remove time from Date

# Display and Plot Future Predictions
st.dataframe(future_df.style.format({'Predicted Price': '{:.2f}'}))
st.markdown("**Future Predicted Prices**")
st.line_chart(future_df.set_index('Date'))

# Footer
st.markdown(
    """
    ---
    #### Created by Akshaya, Abhishek, Sharon, Sudha, Rohan
    """
)
