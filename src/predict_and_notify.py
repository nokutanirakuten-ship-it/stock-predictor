import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from lightgbm import LGBMClassifier
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib

# Optunaのログ出力を抑制してGitHub Actionsのログを綺麗に保つ
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_PATH = "data/model.pkl"
LOG_PATH = "data/history_log.csv"

# 予測ターゲットの閾値（0.005 = 翌日+0.5%以上の上昇を正例とする）
RISE_THRESHOLD = 0.005

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

def dynamic_stock_screening():
    # 候補となる代表的な優良株のプール（100株あたり20万〜30万円レンジを自動スクリーニング）
    candidate_tickers = {
        "7203.T": "トヨタ自動車", "6758.T": "ソニーグループ", 
        "1928.T": "積水ハウス", "5401.T": "日本製鉄", 
        "8411.T": "みずほフィナンシャルグループ", "2503.T": "キリンホールディングス", 
        "8031.T": "三井物産", "8058.T": "三菱商事", 
        "9201.T": "日本航空 (JAL)", "4901.T": "富士フイルムホールディングス",
        "6501.T": "日立製作所", "8306.T": "三菱UFJフィナンシャル・グループ"
    }
    
    selected_dict = {}
    for ticker, name in candidate_tickers.items():
        try:
            df = yf.download(ticker, period="5d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            latest_close = df['Close'].iloc[-1]
            
            # 株価が2,000円〜3,000円の範囲内か判定
            if 2000 <= latest_close <= 3000:
                selected_dict[ticker] = name
        except Exception:
            continue
            
    if len(selected_dict) == 0:
        selected_dict = {"7203.T": "トヨタ自動車", "6758.T": "ソニーグループ"}
        
    return dict(list(selected_dict.items())[:10])

def calculate_technical_indicators(df):
    # テクニカル指標
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

    # 需給・ボラティリティ指標（出来高急増検知・日中値幅率）
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA_20'] + 1e-5) # 出来高倍率（需給の変化）
    df['ATR'] = (df['High'] - df['Low']) / df['Close']          # ボラティリティ指標

    return df

def fetch_macro_data():
    """米国株・為替に加え、国内マクロ（日経平均）とボラティリティ（VIX）を取得"""
    macro_dfs = {}
    tickers_macro = {
        "USDJPY": "USDJPY=X",
        "SP500": "^GSPC",
        "US10Y": "^TNX",
        "N225": "^N225",  # 国内マクロ（日経平均）
        "VIX": "^VIX"      # ボラティリティ指標（恐怖指数）
    }
    for key, t_symbol in tickers_macro.items():
        try:
            df = yf.download(t_symbol, period="1y", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if key == "VIX":
                macro_dfs["VIX_Close"] = df['Close']
                macro_dfs["VIX_Return"] = df['Close'].pct_change()
            else:
                macro_dfs[f"{key}_Return"] = df['Close'].pct_change()
        except Exception:
            pass
    return pd.DataFrame(macro_dfs)

def fetch_ticker_fundamentals(ticker):
    """銘柄ごとのファンダメンタルズ指標（PER, PBR, 配当利回り）を取得"""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        pe = info.get('forwardPE') or info.get('trailingPE') or 0.0
        pbr = info.get('priceToBook') or 0.0
        div_yield = info.get('dividendYield') or 0.0
        return float(pe), float(pbr), float(div_yield)
    except Exception:
        return 0.0, 0.0, 0.0

def optimize_hyperparameters(X, y):
    """Optunaによる時系列交差検証ベースのハイパーパラメータ自動最適化"""
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    return best_params

def run_morning_prediction():
    os.makedirs("data", exist_ok=True)
    
    TICKER_DICT = dynamic_stock_screening()
    
    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
    else:
        df_log = pd.DataFrame(columns=["Date", "Ticker", "Predicted", "Actual", "Result", "Reason"])

    error_penalties = {}
    if not df_log.empty and 'Result' in df_log.columns:
        for ticker in TICKER_DICT.keys():
            t_df = df_log[df_log['Ticker'] == ticker].dropna(subset=['Result'])
            if len(t_df) > 0:
                failure_rate = (t_df['Result'] == "外れ...").sum() / len(t_df)
                error_penalties[ticker] = failure_rate
            else:
                error_penalties[ticker] = 0.0

    df_macro = fetch_macro_data()

    dfs = []
    for ticker in TICKER_DICT.keys():
        df = yf.download(ticker, period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = calculate_technical_indicators(df)
        df['Return'] = df['Close'].pct_change()
        
        # 目的変数（翌日+0.5%以上上昇で正例）
        df['Target'] = (df['Return'].shift(-1) >= RISE_THRESHOLD).astype(int)
        
        # マクロ・ボラティリティデータの結合
        if not df_macro.empty:
            df = df.join(df_macro, how='left')
            
        # ファンダメンタルズ指標の追加
        pe, pbr, div_yield = fetch_ticker_fundamentals(ticker)
        df['PER'] = pe
        df['PBR'] = pbr
        df['Div_Yield'] = div_yield

        df['Historical_Error_Rate'] = error_penalties.get(ticker, 0.0)
        df['Ticker'] = ticker
        dfs.append(df)
    
    df_all = pd.concat(dfs).dropna()
    
    # 特徴量カラムの構成
    feature_cols = [
        'SMA_5', 'SMA_20', 'BB_High', 'BB_Low', 'RSI', 'MACD', 'Return',
        'Vol_Ratio', 'ATR', 'PER', 'PBR', 'Div_Yield', 'Historical_Error_Rate'
    ]
    for macro_col in ['USDJPY_Return', 'SP500_Return', 'US10Y_Return', 'N225_Return', 'VIX_Close', 'VIX_Return']:
        if macro_col in df_all.columns:
            feature_cols.append(macro_col)
        
    X = df_all[feature_cols]
    y = df_all['Target']
    
    # Optunaによる自動ハイパーパラメータ最適化とモデル構築
    best_params = optimize_hyperparameters(X, y)
    model = LGBMClassifier(**best_params)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    today = datetime.now().strftime("%Y-%m-%d")
    predictions = []
    mail_body = f"【フルスペック高度株価予測レポート】({today})\n\n国内マクロ・VIX・需給・ファンダメンタルズ指標をフル統合し、Optuna最適化を適用した本日の予測です。\n\n"

    for ticker, name in TICKER_DICT.items():
        latest_data = df_all[df_all['Ticker'] == ticker].tail(1)
        if not latest_data.empty:
            pred = int(model.predict(latest_data[feature_cols])[0])
            pred_text = "明確な上昇期待 (+0.5%以上)" if pred == 1 else "横ばい・下落懸念"
            
            latest_rsi = latest_data['RSI'].values[0]
            latest_vol_ratio = latest_data['Vol_Ratio'].values[0]
            reason_hint = f"RSI({latest_rsi:.1f})・出来高倍率({latest_vol_ratio:.1f}倍)・マクロ/VIX/ファンダメンタル統合判断"
            
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

    send_email(f"【朝の株価予測レポート】({today})", mail_body)

if __name__ == "__main__":
    run_morning_prediction()
