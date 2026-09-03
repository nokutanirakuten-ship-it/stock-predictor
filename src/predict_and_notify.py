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
    
    # 1. 蓄積された過去の履歴ログを読み込み
    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
    else:
        df_log = pd.DataFrame(columns=["Date", "Ticker", "Predicted", "Actual", "Result"])

    # 2. 履歴データから銘柄ごとの過去の正解率（フィードバック指標）を計算
    ticker_success_rates = {}
    if not df_log.empty and 'Result' in df_log.columns:
        # 「当たり！」の数をカウントして成功率を算出
        for ticker in TICKER_DICT.keys():
            t_df = df_log[df_log['Ticker'] == ticker]
            valid_t_df = t_df.dropna(subset=['Result'])
            if len(valid_t_df) > 0:
                success_count = (valid_t_df['Result'] == "当たり！").sum()
                ticker_success_rates[ticker] = success_count / len(valid_t_df)
            else:
                ticker_success_rates[ticker] = 0.5  / デフォルト値

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
        df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
        
        # 過去の履歴から得られた正解率を特徴量として各行に付与（自己フィードバック）
        df['Historical_Success_Rate'] = ticker_success_rates.get(ticker, 0.5)
        
        if usdjpy_series is not None:
            df['USDJPY_Return'] = usdjpy_series
            
        df['Ticker'] = ticker
        dfs.append(df)
    
    df_all = pd.concat(dfs).dropna()
    
    # 訓練に使う特徴量リストに履歴フィードバック項目を追加
    feature_cols = ['SMA_5', 'SMA_20', 'BB_High', 'BB_Low', 'RSI', 'MACD', 'Return', 'Historical_Success_Rate']
    if 'USDJPY_Return' in df_all.columns:
        feature_cols.append('USDJPY_Return')
        
    X = df_all[feature_cols]
    y = df_all['Target']
    
    # モデルの学習
    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    today = datetime.now().strftime("%Y-%m-%d")
    predictions = []
    mail_body = f"【自己学習型・株価予測レポート】({today})\n\n過去の検証履歴（ログ）を学習にフィードバックした本日の予測です。\n\n"

    for ticker, name in TICKER_DICT.items():
        latest_data = df_all[df_all['Ticker'] == ticker].tail(1)
        if not latest_data.empty:
            pred = int(model.predict(latest_data[feature_cols])[0])
            pred_text = "上がりそう (1)" if pred == 1 else "下がりそう (0)"
            mail_body += f"・銘柄: {name} ({ticker}) -> 予測: {pred_text}\n"
            predictions.append({"Date": today, "Ticker": ticker, "Predicted": pred, "Actual": None, "Result": None})

    df_new_log = pd.concat([df_log, pd.DataFrame(predictions)], ignore_index=True)
    df_new_log.to_csv(LOG_PATH, index=False)

    send_email(f"【朝の自己学習型予測】{today}", mail_body)

if __name__ == "__main__":
    run_morning_prediction()
