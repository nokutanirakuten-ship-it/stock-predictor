import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from lightgbm import LGBMClassifier
import joblib

# 銘柄コードと日本語銘柄名のマッピング（10銘柄）
TICKER_DICT = {
    "7203.T": "トヨタ自動車",
    "6758.T": "ソニーグループ",
    "1928.T": "積水ハウス",
    "5401.T": "日本製鉄",
    "8411.T": "みずほフィナンシャルグループ",
    "2503.T": "キリンホールディングス",
    "8031.T": "三井物産",
    "8058.T": "三菱商事",
    "9201.T": "日本航空 (JAL)",
    "4901.T": "富士フイルムホールディングス"
}

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

def calculate_technical_indicators(df):
    # 移動平均線
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # ボリンジャーバンド (20日)
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_High'] = df['SMA_20'] + (std_20 * 2)
    df['BB_Low'] = df['SMA_20'] - (std_20 * 2)
    
    # RSI (14日)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    return df

def run_morning_prediction():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
    else:
        df_log = pd.DataFrame(columns=["Date", "Ticker", "Predicted", "Actual", "Result"])

    # マクロ指標（ドル円）の取得
    try:
        df_usdjpy = yf.download("USDJPY=X", period="1y", progress=False)
        if isinstance(df_usdjpy.columns, pd.MultiIndex):
            df_usdjpy.columns = df_usdjpy.columns.get_level_values(0)
        df_usdjpy['USDJPY_Return'] = df_usdjpy['Close'].pct_change()
        usdjpy_series = df_usdjpy['USDJPY_Return']
    except Exception:
        usdjpy_series = None

    dfs = []
    for ticker in TICKER_DICT.keys():
        df = yf.download(ticker, period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = calculate_technical_indicators(df)
        df['Return'] = df['Close'].pct_change()
        
        # ターゲットを「翌日の終値が当日より高いか（今日の予測）」に戻す
        df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
        
        if usdjpy_series is not None:
            df['USDJPY_Return'] = usdjpy_series
            
        df['Ticker'] = ticker
        dfs.append(df)
    
    df_all = pd.concat(dfs).dropna()
    
    feature_cols = ['SMA_5', 'SMA_20', 'BB_High', 'BB_Low', 'RSI', 'MACD', 'Return']
    if 'USDJPY_Return' in df_all.columns:
        feature_cols.append('USDJPY_Return')
        
    X = df_all[feature_cols]
    y = df_all['Target']
    
    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    today = datetime.now().strftime("%Y-%m-%d")
    predictions = []
    mail_body = f"【株価予測レポート】({today})\n\nテクニカル指標・為替連動を反映した本日の予測です。\n\n"

    for ticker, name in TICKER_DICT.items():
        latest_data = df_all[df_all['Ticker'] == ticker].tail(1)
        if not latest_data.empty:
            pred = int(model.predict(latest_data[feature_cols])[0])
            pred_text = "上がりそう (1)" if pred == 1 else "下がりそう (0)"
            mail_body += f"・銘柄: {name} ({ticker}) -> 予測: {pred_text}\n"
            predictions.append({"Date": today, "Ticker": ticker, "Predicted": pred, "Actual": None, "Result": None})

    df_new_log = pd.concat([df_log, pd.DataFrame(predictions)], ignore_index=True)
    df_new_log.to_csv(LOG_PATH, index=False)

    send_email(f"【朝の株価予測】{today}", mail_body)

if __name__ == "__main__":
    run_morning_prediction()
