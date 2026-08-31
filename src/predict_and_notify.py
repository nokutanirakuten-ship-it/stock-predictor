import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pandas as pd
import yfinance as yf
from lightgbm import LGBMClassifier
import joblib

# 100株あたり20万〜30万円前後の優良日本株10銘柄
TICKERS = [
    "7203.T",  # トヨタ自動車
    "6758.T",  # ソニーグループ
    "1928.T",  # 積水ハウス
    "5401.T",  # 日本製鉄
    "8411.T",  # みずほフィナンシャルグループ
    "2503.T",  # キリンホールディングス
    "8031.T",  # 三井物産
    "8058.T",  # 三菱商事
    "9201.T",  # 日本航空 (JAL)
    "4901.T"   # 富士フイルムホールディングス
]

MODEL_PATH = "data/model.pkl"
LOG_PATH = "data/history_log.csv"

def send_email(subject, body):
    sender_email = os.environ.get("MAIL_USER")
    app_password = os.environ.get("MAIL_PASS")
    receiver_email = "nokutani.rakuten@gmail.com"

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, app_password)
        server.send_message(msg)

def run_morning_prediction():
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

    today = datetime.now().strftime("%Y-%m-%d")
    predictions = []
    mail_body = f"【株価予測レポート】({today})\n\n"

    for ticker in TICKERS:
        latest_data = df_all[df_all['Ticker'] == ticker].tail(1)
        if not latest_data.empty:
            pred = int(model.predict(latest_data[['SMA_5', 'Return']])[0])
            pred_text = "上がりそう (1)" if pred == 1 else "下がりそう (0)"
            mail_body += f"・銘柄: {ticker} -> 予測: {pred_text}\n"
            predictions.append({"Date": today, "Ticker": ticker, "Predicted": pred, "Actual": None, "Result": None})

    df_new_log = pd.concat([df_log, pd.DataFrame(predictions)], ignore_index=True)
    df_new_log.to_csv(LOG_PATH, index=False)

    send_email(f"【朝の株価予測】{today}", mail_body)

if __name__ == "__main__":
    run_morning_prediction()
