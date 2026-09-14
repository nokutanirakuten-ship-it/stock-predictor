import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pandas as pd
import yfinance as yf

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
    mail_body = f"【検証・要因分析レポート】({today})\n\n"

    updated_rows = []
    for index, row in df_log.iterrows():
        if row['Date'] == today and pd.isna(row['Actual']):
            ticker = row['Ticker']
            name = TICKER_DICT.get(ticker, ticker)
            predicted = row['Predicted']
            morning_reason = row.get('Reason', '特になし')
            
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
                
                # 当たり・外れの理由を明確化
                if result == "当たり！":
                    detailed_reason = f"【的中理由】朝の予測根拠（{morning_reason}）通りに、本日の株価が想定通りの方向へ推移しました。"
                else:
                    detailed_reason = f"【外れた原因】朝は（{morning_reason}）と予測しましたが、予想外の地合いの変化や個別要因により実際の株価は逆方向に動きました。"
                
                row['Reason'] = detailed_reason
                mail_body += f"・銘柄: {name}\n  予測: {predicted} / 実際: {actual} -> 【 {result} 】\n  {detailed_reason}\n\n"
        updated_rows.append(row)

    df_updated = pd.DataFrame(updated_rows)
    df_updated.to_csv(LOG_PATH, index=False)

    send_email(f"【夕方の検証・要因分析】{today}", mail_body)

if __name__ == "__main__":
    run_evening_verification()
