import os
import pandas as pd
import yfinance as yf
from lightgbm import LGBMClassifier
import joblib
from datetime import datetime

TICKERS = ["7203.T", "9984.T"]
MODEL_PATH = "data/model.pkl"
LOG_PATH = "data/history_log.csv"

def run_pipeline():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
    else:
        df_log = pd.DataFrame(columns=["Date", "Ticker", "Predicted", "Actual", "Result"])

    dfs = []
    for ticker in TICKERS:
        df = yf.download(ticker, period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['Ticker'] = ticker
        dfs.append(df)

    df_all = pd.concat(dfs).dropna()
    X = df_all[['SMA_5', 'Return']]
    y = df_all['Target']

    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    predictions = []
    today = datetime.now().strftime("%Y-%m-%d")
    for ticker in TICKERS:
        latest_data = df_all[df_all['Ticker'] == ticker].tail(1)
        if not latest_data.empty:
            pred = model.predict(latest_data[['SMA_5', 'Return']])[0]
            predictions.append({"Date": today, "Ticker": ticker, "Predicted": int(pred), "Actual": None, "Result": None})

    df_new_log = pd.concat([df_log, pd.DataFrame(predictions)], ignore_index=True)
    df_new_log.to_csv(LOG_PATH, index=False)
    print("Pipeline executed successfully.")

if __name__ == "__main__":
    run_pipeline()
