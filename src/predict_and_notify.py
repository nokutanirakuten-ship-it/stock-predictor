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
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_High'] = df['SMA_20'] + (std_20 * 2)
    df['BB_Low'] = df['SMA_20'] - (std_20 * 2)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    return df

def fetch_macro_data():
    """為替、米国株（S&P500）、米国債利回りのデータを取得"""
    macro_dfs = {}
    tickers_macro = {
        "USDJPY": "USDJPY=X",
        "SP500": "^GSPC",
        "US10Y": "^TNX"
    }
    for key, t_symbol in tickers_macro.items():
        try:
            df = yf.download(t_symbol, period="1y", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            macro_dfs[f"{key}_Return"] = df['Close'].pct_change()
        except Exception:
            pass
    return pd.DataFrame(macro_dfs)

def run_morning_prediction():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
    else:
        df_log = pd.DataFrame(columns=["Date", "Ticker", "Predicted", "Actual", "Result", "Reason"])

    # 過去の外れパターンから「ペナルティ補正値」を算出
    error_penalties = {}
    if not df_log.empty and 'Result' in df_log.columns:
        for ticker in TICKER_DICT.keys():
            t_df = df_log[df_log['Ticker'] == ticker].dropna(subset=['Result'])
            if len(t_df) > 0:
                failure_rate = (t_df['Result'] == "外れ...").sum() / len(t_df)
                error_penalties[ticker] = failure_rate # 外れ率が高いほど慎重に補正
            else:
                error_penalties[ticker] = 0.0

    # マクロ経済データの取得
    df_macro = fetch_macro_data()

    dfs = []
    for ticker in TICKER_DICT.keys():
        df = yf.download(ticker, period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = calculate_technical_indicators(df)
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
        
        # マクロデータを結合
        if not df_macro.empty:
            df = df.join(df_macro, how='left')
            
        # 過去の外れ率を特徴量（リスクファクター）として付与
        df['Historical_Error_Rate'] = error_penalties.get(ticker, 0.0)
        
        df['Ticker'] = ticker
        dfs.append(df)
    
    df_all = pd.concat(dfs).dropna()
    
    # 特徴量カラムの定義（テクニカル ＋ 為替・米株・金利 ＋ エラー履歴）
    feature_cols = ['SMA_5', 'SMA_20', 'BB_High', 'BB_Low', 'RSI', 'MACD', 'Return', 'Historical_Error_Rate']
    for macro_col in ['USDJPY_Return', 'SP500_Return', 'US10Y_Return']:
        if macro_col in df_all.columns:
            feature_cols.append(macro_col)
        
    X = df_all[feature_cols]
    y = df_all['Target']
    
    # モデル学習
    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    today = datetime.now().strftime("%Y-%m-%d")
    predictions = []
    mail_body = f"【経済指標統合・自己学習型予測】({today})\n\n米国株・金利・為替および過去の外れ要因を分析に反映した本日の予測です。\n\n"

    for ticker, name in TICKER_DICT.items():
        latest_data = df_all[df_all['Ticker'] == ticker].tail(1)
        if not latest_data.empty:
            pred = int(model.predict(latest_data[feature_cols])[0])
            pred_text = "上がりそう (1)" if pred == 1 else "下がりそう (0)"
            
            # 予測の根拠となる主要因を簡易判定
            latest_rsi = latest_data['RSI'].values[0]
            macro_trend = "米株・為替連動"
            reason_hint = f"RSI({latest_rsi:.1f})と{macro_trend}の傾向から判断"
            
            mail_body += f"・銘柄: {name} ({ticker}) -> 予測: {pred_text}\n  (根拠: {reason_hint})\n\n"
            predictions.append({
                "Date": today, 
                "Ticker": ticker, 
                "Predicted": pred, 
                "Actual": None, 
                "Result": None, 
                "Reason": reason_hint
            })

    df_new_log = pd.concat([df_log, pd.DataFrame(predictions)], ignore_index=True)
    df_new_log.to_csv(LOG_PATH, index=False)

    send_email(f"【朝の経済指標・自己学習予測】{today}", mail_body)

if __name__ == "__main__":
    run_morning_prediction()
