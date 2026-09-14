import os
import pandas as pd
import streamlit as st

# 1. ページ基本設定（モバイル・PWA表示に最適化）
st.set_page_config(
    page_title="株価予測AIダッシュボード",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# PWA風全画面表示用のメタタグ挿入（タイポ修正箇所）
st.markdown("""
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
""", unsafe_allow_html=True)

# 2. ヘッダー表示
st.title("📈 日本株 AI予測ダッシュボード")
st.caption("LightGBM + Optuna + Tree SHAP による翌日株価予測システム")

LOG_PATH = "data/history_log.csv"

# 銘柄名マッピング
TICKER_NAMES = {
    "7203.T": "トヨタ自動車",
    "6758.T": "ソニーグループ",
    "1928.T": "積水ハウス",
    "5401.T": "日本製鉄",
    "8411.T": "みずほFG",
    "2503.T": "キリンHD",
    "8031.T": "三井物産",
    "8058.T": "三菱商事",
    "9201.T": "日本航空 (JAL)",
    "4901.T": "富士フイルムHD",
    "6501.T": "日立製作所",
    "8306.T": "三菱UFJ"
}

# 3. データ読み込みとダッシュボード構築
if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)
    
    if not df_log.empty:
        # 銘柄名の補完
        df_log['Ticker_Name'] = df_log['Ticker'].map(lambda x: f"{TICKER_NAMES.get(x, x)} ({x})")
        
        # --- パフォーマンス概要（メトリクス） ---
        st.subheader("📊 予測パフォーマンス概要")
        
        df_evaluated = df_log.dropna(subset=['Result'])
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("累計予測件数", f"{len(df_log)} 件")
        with col2:
            if not df_evaluated.empty:
                win_count = (df_evaluated['Result'] == "的中！").sum()
                win_rate = (win_count / len(df_evaluated)) * 100
                st.metric("予測的中率", f"{win_rate:.1f} %")
            else:
                st.metric("予測的中率", "検証中...")
        with col3:
            latest_date = df_log['Date'].max()
            st.metric("最新予測日", str(latest_date))
            
        st.divider()

        # --- 最新の予測結果 ---
        st.subheader(f"📅 最新予測結果 ({latest_date})")
        df_latest = df_log[df_log['Date'] == latest_date].copy()
        
        for _, row in df_latest.iterrows():
            is_up = row['Predicted'] == 1
            pred_label = "🚀 明確な上昇期待 (+0.5%以上)" if is_up else "➡️ 横ばい・下落懸念"
            
            with st.expander(f"**{row['Ticker_Name']}** ： {pred_label}", expanded=True):
                st.write(f"**判定理由 (SHAP分析)**: {row.get('Reason', '解析中')}")
                if pd.notna(row.get('Result')):
                    st.write(f"**検証結果**: {row['Result']} (実績騰落: {row.get('Actual', '-')})")

        st.divider()

        # --- 予測・検証履歴ログ ---
        st.subheader("📜 過去の予測ログ")
        
        tickers_list = ["すべて"] + list(df_log['Ticker_Name'].unique())
        selected_ticker = st.selectbox("銘柄絞り込み", tickers_list)
        
        if selected_ticker != "すべて":
            display_df = df_log[df_log['Ticker_Name'] == selected_ticker]
        else:
            display_df = df_log
            
        display_cols = ["Date", "Ticker_Name", "Predicted", "Actual", "Result", "Reason"]
        existing_cols = [c for c in display_cols if c in display_df.columns]
        
        st.dataframe(
            display_df[existing_cols].sort_values(by="Date", ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("予測履歴データ (`data/history_log.csv`) は存在しますが、まだデータが記録されていません。")
else:
    st.warning("⚠️ 予測履歴データ (`data/history_log.csv`) がまだ見つかりません。")
    st.info("GitHub Actions による朝の自動予測処理が完了すると、ここに最新結果が表示されます。")
