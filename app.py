import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="株価予測ダッシュボード", layout="wide")

st.title("📈 自動株価予測・検証ダッシュボード")

LOG_PATH = "data/history_log.csv"

if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)
    
    # データの基本統計
    total_preds = len(df_log)
    valid_logs = df_log.dropna(subset=['Result'])
    total_verified = len(valid_logs)
    
    if total_verified > 0:
        wins = (valid_logs['Result'] == "当たり！").sum()
        win_rate = (wins / total_verified) * 100
    else:
        win_rate = 0.0

    # メトリクス表示
    col1, col2, col3 = st.columns(3)
    col1.metric("総予測回数", f"{total_preds} 回")
    col2.metric("検証済み回数", f"{total_verified} 回")
    col3.metric("直近勝率", f"{win_rate:.1f}%")

    st.markdown("---")
    st.subheader("📋 予測・検証履歴ログ一覧")
    
    # フィルター機能
    ticker_filter = st.selectbox("銘柄で絞り込み", ["すべて"] + list(df_log['Ticker'].unique()))
    if ticker_filter != "すべて":
        filtered_df = df_log[df_log['Ticker'] == ticker_filter]
    else:
        filtered_df = df_log

    st.dataframe(filtered_df.sort_values(by="Date", ascending=False), use_container_width=True)

else:
    st.warning("まだ予測履歴ログ（history_log.csv）が生成されていません。朝の予測プログラムの実行をお待ちください。")
