# StockAnalysis
import streamlit as st  # For creating web application
import yfinance as yf  # For fetching stock data
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced visualizations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

# Configure plot style
plt.style.use('seaborn-v0_8-bright')


@st.cache_data
def fetch_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    
    if data.empty:
        st.error("No stock data found. Check ticker symbols or date range.")
        return None  # Prevents KeyError

    # Handle MultiIndex DataFrame
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Adj Close'] if 'Adj Close' in data else data['Close']
    else:
        data = data[['Adj Close']] if 'Adj Close' in data else data[['Close']]
    
    return data




# Fetch fundamental metrics using yfinance
def fetch_fundamental_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "P/B Ratio": info.get("priceToBook", "N/A"),
            "ROE": info.get("returnOnEquity", "N/A"),
            "Current Ratio": info.get("currentRatio", "N/A"),
            "Debt-to-Equity": info.get("debtToEquity", "N/A"),
            "Profit Margin": info.get("profitMargins", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "Beta": info.get("beta", "N/A"),
        }
    except Exception as e:
        return {"Error": str(e)}

# Placeholder for advanced sentiment analysis
def sentiment_analysis(ticker):
    sample_news = [
        f"{ticker} shows promising growth this quarter",
        f"{ticker} faces challenges due to market conditions",
    ]
    sentiment_scores = []
    for news in sample_news:
        if "promising" in news.lower():
            sentiment_scores.append(1)
        elif "challenges" in news.lower():
            sentiment_scores.append(-1)
        else:
            sentiment_scores.append(0)
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else "No News Available"
    return avg_sentiment

# Display fundamental metrics for selected stocks
def display_fundamentals(tickers):
    st.subheader("Fundamental Metrics")
    metrics = {}
    for ticker in tickers:
        metrics[ticker] = fetch_fundamental_metrics(ticker)
    st.write(pd.DataFrame(metrics).T)

# Sentiment analysis display
def display_sentiment(tickers):
    st.subheader("Sentiment Analysis")
    sentiments = {}
    for ticker in tickers:
        sentiments[ticker] = sentiment_analysis(ticker)
    st.write(pd.DataFrame.from_dict(sentiments, orient='index', columns=['Sentiment Score']))

# Plot and display time series of closing prices
def time_series_analysis(df):
    st.subheader("Time Series Analysis: Closing Prices")
    plt.figure(figsize=(12, 6))
    df.plot(title='Time Series Analysis: Closing Prices', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Adjusted Closing Price')
    plt.legend(df.columns)
    st.pyplot(plt)

# Calculate and display annualized volatility of stocks
def volatility_analysis(df):
    st.subheader("Volatility Analysis")
    volatility = df.pct_change().std() * np.sqrt(252)
    st.write(volatility)
    plt.figure(figsize=(8, 4))
    volatility.plot(kind='bar', color='skyblue')
    plt.title('Stock Volatility (Annualized)')
    plt.xlabel('Stock')
    plt.ylabel('Volatility')
    st.pyplot(plt)

# Display and visualize the correlation matrix
def correlation_analysis(df):
    st.subheader("Correlation Analysis")
    correlation = df.corr()
    st.write(correlation)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Stocks')
    st.pyplot(plt)

# Compare cumulative returns of stocks
def comparative_analysis(df):
    st.subheader("Comparative Analysis")
    cumulative_returns = (df / df.iloc[0]) * 100
    plt.figure(figsize=(12, 6))
    cumulative_returns.plot(title='Comparative Analysis: Normalized Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend(df.columns)
    st.pyplot(plt)

# Analyze and visualize risk-return tradeoff
def risk_return_tradeoff(df):
    st.subheader("Risk-Return Tradeoff Analysis")
    returns = df.pct_change().mean() * 252
    risks = df.pct_change().std() * np.sqrt(252)

    risk_return_df = pd.DataFrame({'Return': returns, 'Risk': risks})

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=risk_return_df, x='Risk', y='Return', s=100, color='purple')
    for i in risk_return_df.index:
        plt.text(risk_return_df.Risk[i], risk_return_df.Return[i], i)
    plt.title('Risk-Return Tradeoff Analysis')
    plt.xlabel('Risk (Volatility)')
    plt.ylabel('Return')
    plt.grid(True)
    st.pyplot(plt)

# Predict future stock prices
def predictive_modeling(df, stock):
    st.subheader(f"Predictive Modeling for {stock}")
    data = df[[stock]].dropna()
    data['Days'] = range(len(data))
    X = data['Days'].values.reshape(-1, 1)
    y = data[stock].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(data['Days'], data[stock], label='Actual Prices')
    plt.scatter(X_test, predictions, color='red', label='Predicted Prices')
    plt.title(f"{stock} Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

# Portfolio optimization using Modern Portfolio Theory
def portfolio_optimization(df):
    st.subheader("Portfolio Optimization")
    returns = df.pct_change().mean() * 252
    cov_matrix = df.pct_change().cov() * 252

    num_assets = len(df.columns)

    def objective(weights):
        portfolio_return = np.sum(weights * returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1.0 / num_assets]

    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    st.write(f"Optimal Weights: {dict(zip(df.columns, optimal_weights))}")

    plt.figure(figsize=(10, 6))
    plt.bar(df.columns, optimal_weights, color='green')
    plt.title("Optimal Portfolio Weights")
    plt.xlabel("Stocks")
    plt.ylabel("Weight")
    st.pyplot(plt)

# Main Streamlit app function
def main():
    st.title("📈 Stock Analysis Web Application")

    st.sidebar.title("🔍 Options")
    tickers = st.sidebar.text_input("Enter stock tickers (space-separated, e.g., AAPL MSFT GOOGL):")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))

    if tickers:
        tickers_list = tickers.upper().split()
        stock_data = fetch_stock_data(tickers_list, start_date=start_date, end_date=end_date)

        if stock_data.empty:
            st.error("No data found for the given tickers. Please try again.")
        else:
            st.download_button(
                "Download Stock Data as CSV",
                stock_data.to_csv().encode('utf-8'),
                "stock_data.csv",
                "text/csv",
            )

            st.subheader("Descriptive Statistics")
            st.write(stock_data.describe())

            time_series_analysis(stock_data)
            volatility_analysis(stock_data)
            correlation_analysis(stock_data)
            comparative_analysis(stock_data)
            risk_return_tradeoff(stock_data)

            st.markdown("---")
            display_fundamentals(tickers_list)

            st.markdown("---")
            display_sentiment(tickers_list)

            st.markdown("---")
            stock_for_prediction = st.selectbox("Select a stock for predictive modeling:", tickers_list)
            if stock_for_prediction:
                predictive_modeling(stock_data, stock_for_prediction)

            if len(tickers_list) > 1:
                portfolio_optimization(stock_data)

# Entry point of the program
if __name__ == "__main__":
    main()
  
