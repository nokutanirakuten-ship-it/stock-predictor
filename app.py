import os
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# 1. ページ基本設定 & モバイル/PWA向けCSS最適化
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI株価予測",
    page_icon="📈",
    layout="centered",  # スマホ画面に最適なセンタリング表示
    initial_sidebar_state="collapsed"
)

# モダンアプリ風のカスタムスタイリング
st.markdown("""
    <style>
    /* 全体コンテナの幅調整 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 680px;
    }
    
    /* カード型コンテナ */
    .stock-card {
        background: #ffffff;
        border: 1px solid #edf2f7;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    
    /* 上昇・下落バッジ */
    .badge-up {
        background-color: #e6f4ea;
        color: #137333;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-down {
        background-color: #f1f3f4;
        color: #5f6368;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* SHAP理由ボックス */
    .reason-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4285f4;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #3c4043;
        border-radius: 0 8px 8px 0;
        margin-top: 10px;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

LOG_PATH = "data/history_log.csv"

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

# ---------------------------------------------------------
# 2. アプリヘッダー
# ---------------------------------------------------------
st.title("📈 AI株価予測")
st.caption("LightGBM × Optuna × SHAP 解析モデル")

if not os.path.exists(LOG_PATH):
    st.warning("⚠️ 予測データが見つかりません。")
    st.info("毎朝の自動バッチ処理が完了すると、最新データがここに反映されます。")
    st.stop()

df_log = pd.read_csv(LOG_PATH)
if df_log.empty:
    st.info("現在ログデータは空です。")
    st.stop()

df_log['Ticker_Name'] = df_log['Ticker'].map(lambda x: TICKER_NAMES.get(x, x))
latest_date = df_log['Date'].max()

# ---------------------------------------------------------
# 3. タブ型ナビゲーション（UX最適化）
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔥 本日の予測", "📊 勝率・分析", "📜 過去ログ"])

# --- TAB 1: 本日の予測 ---
with tab1:
    st.caption(f"最終更新: {latest_date}")
    df_latest = df_log[df_log['Date'] == latest_date].copy()
    
    # 注目（上昇予測）の銘柄数をカウント
    up_count = (df_latest['Predicted'] == 1).sum()
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("本日シグナル対象", f"{len(df_latest)} 銘柄")
    with col_b:
        st.metric("上昇期待銘柄", f"{up_count} 銘柄")
        
    st.markdown("---")
    
    for _, row in df_latest.iterrows():
        is_up = row['Predicted'] == 1
        badge_html = '<span class="badge-up">🚀 上昇期待 (+0.5%↑)</span>' if is_up else '<span class="badge-down">➡️ 静観 / 静かな推移</span>'
        reason = row.get('Reason', '解析データなし')
        
        st.markdown(f"""
            <div class="stock-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #1a73e8;">
                        {row['Ticker_Name']} <span style="font-size: 0.8rem; color: #70757a; font-weight: normal;">({row['Ticker']})</span>
                    </h3>
                    {badge_html}
                </div>
                <div class="reason-box">
                    <strong>💡 判断根拠:</strong><br>{reason}
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- TAB 2: 勝率・分析 ---
with tab2:
    st.subheader("🎯 モデル精度評価")
    df_eval = df_log.dropna(subset=['Result']).copy()
    
    if df_eval.empty:
        st.info("※ 予測結果の答え合わせデータが集計されるまでお待ちください。")
    else:
        win_count = (df_eval['Result'] == "的中！").sum()
        total_count = len(df_eval)
        win_rate = (win_count / total_count) * 100
        
        m1, m2, m3 = st.columns(3)
        m1.metric("総検証数", f"{total_count} 件")
        m2.metric("的中数", f"{win_count} 件")
        m3.metric("通算勝率", f"{win_rate:.1f} %")
        
        st.markdown("---")
        st.write("**銘柄別の勝率内訳**")
        
        ticker_stats = []
        for ticker, name in TICKER_NAMES.items():
            t_df = df_eval[df_eval['Ticker'] == ticker]
            if len(t_df) > 0:
                t_wins = (t_df['Result'] == "的中！").sum()
                t_rate = (t_wins / len(t_df)) * 100
                ticker_stats.append({
                    "銘柄名": name,
                    "コード": ticker,
                    "検証数": len(t_df),
                    "的中率": f"{t_rate:.1f}%"
                })
        
        if ticker_stats:
            st.dataframe(pd.DataFrame(ticker_stats), use_container_width=True, hide_index=True)

# --- TAB 3: 過去ログ ---
with tab3:
    st.subheader("📜 予測・結果データ一覧")
    
    # フィルターUI
    selected_ticker = st.selectbox(
        "銘柄絞り込み", 
        ["すべての銘柄"] + list(df_log['Ticker_Name'].unique())
    )
    
    df_display = df_log.copy()
    if selected_ticker != "すべての銘柄":
        df_display = df_display[df_display['Ticker_Name'] == selected_ticker]
        
    # テーブル表示用の整形
    df_display['予測'] = df_display['Predicted'].map({1: "🚀 上昇", 0: "➡️ 静観"})
    df_table = df_display[['Date', 'Ticker_Name', '予測', 'Actual', 'Result', 'Reason']].sort_values(by="Date", ascending=False)
    
    st.dataframe(
        df_table,
        column_config={
            "Date": "日付",
            "Ticker_Name": "銘柄名",
            "Actual": "実績騰落率",
            "Result": "判定",
            "Reason": "SHAP分析根拠"
        },
        use_container_width=True,
        hide_index=True
    )
