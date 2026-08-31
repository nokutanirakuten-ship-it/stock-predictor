import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pandas as pd
import yfinance as yf

TICKERS = ["7203.T", "9984.T"]
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

def run_evening_verification():
    if not os.path.exists(LOG_PATH):
        return

    df_log = pd.read_csv(LOG_PATH)
    today = datetime.now().strftime("%Y-%m-%d")
    mail_body = f"【株価結果検証レポート】({today})\n\n"

    updated_rows = []
    for index, row in df_log.iterrows():
        if row['Date'] == today and pd.isna(row['Actual']):
            ticker = row['Ticker']
            predicted = row['Predicted']
            
            # 本日の実際の株価データを取得して検証
            df = yf.download(ticker, period="5d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if len(df) >= 2:
                prev_close = df['Close'].iloc[-2]
                curr_close = df['Close'].iloc[-1]
                actual = 1 if curr_close > prev_close else 0
                result = "当たり！" if predicted == actual else "外れ..."
                
                row['Actual'] = actual
                row['Result'] = result
                
                reason = "直近の移動平均線(SMA_5)とリターンの傾向に基づき予測しましたが、本日の実際の相場変動（終値の増減）により結果が分かれました。"
                mail_body += f"・銘柄: {ticker}\n  予測: {predicted} / 実際: {actual} -> 【 {result} 】\n  理由考察: {reason}\n\n"
        updated_rows.append(row)

    df_updated = pd.DataFrame(updated_rows)
    df_updated.to_csv(LOG_PATH, index=False)

    send_email(f"【夕方の検証結果】{today}", mail_body)

if __name__ == "__main__":
    run_evening_verification()
