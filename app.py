import os
import json
import base64
import requests
import pandas as pd
import streamlit as st
import yfinance as yf

# 1. ページ基本設定
st.set_page_config(
    page_title="AI株価予測",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 680px;
    }
    .stock-card {
        background: #ffffff;
        border: 1px solid #edf2f7;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
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
TICKER_JSON_PATH = "data/tickers.json"

def load_tickers():
    if os.path.exists(TICKER_JSON_PATH):
        with open(TICKER_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "7203.T": "トヨタ自動車",
        "6758.T": "ソニーグループ"
    }

def save_tickers(tickers_dict):
    os.makedirs("data", exist_ok=True)
    with open(TICKER_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(tickers_dict, f, ensure_ascii=False, indent=2)
        
    gh_token = st.secrets.get("GH_TOKEN") or os.environ.get("GH_TOKEN")
    repo = st.secrets.get("GH_REPO") or "nokutanirakuten-ship-it/stock-predictor"
    
    if gh_token:
        try:
            url = f"https://api.github.com/repos/{repo}/contents/{TICKER_JSON_PATH}"
            headers = {"Authorization": f"token {gh_token}"}
            res = requests.get(url, headers=headers).json()
            sha = res.get("sha")
            
            content_str = json.dumps(tickers_dict, ensure_ascii=False, indent=2)
            content_b64 = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
            
            payload = {
                "message": "Update tickers.json via Streamlit App",
                "content": content_b64,
                "sha": sha
            }
            requests.put(url, headers=headers, json=payload)
        except Exception:
            pass

def search_stock_candidates(query):
    """入力されたキーワードから日本株の候補リストを取得"""
    candidates = {}
    if not query:
        return candidates
        
    query_str = str(query).strip()
    
    # 4桁数値コード直接入力時のショートカット（例: 7203）
    if query_str.isdigit() and len(query_str) == 4:
        symbol = f"{query_str}.T"
        try:
            info = yf.Ticker(symbol).info
            name = info.get('shortName') or info.get('longName') or symbol
            candidates[symbol] = name
        except Exception:
            candidates[symbol] = f"銘柄コード {symbol}"
            
    # yfinance の Search API によるあいまい検索
    try:
        search_res = yf.Search(query_str, max_results=8)
        quotes = getattr(search_res, 'quotes', [])
        for q in quotes:
            symbol = q.get('symbol', '')
            name = q.get('shortname') or q.get('longname') or symbol
            # 東証銘柄 (.T) のみに絞り込み
            if symbol.endswith('.T'):
                candidates[symbol] = name
    except Exception:
        pass
        
    return candidates

TICKER_NAMES = load_tickers()

@st.cache_data
def get_company_name(ticker):
    if ticker in TICKER_NAMES:
        return TICKER_NAMES[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get('shortName') or info.get('longName') or ticker
        return name
    except Exception:
        return ticker

# 2. アプリヘッダー
st.title("AI株価予測")
st.caption("LightGBM x Optuna x SHAP 解析モデル")

if not os.path.exists(LOG_PATH):
    st.warning("予測データが見つかりません。")
    st.info("毎朝の自動バッチ処理が完了すると、最新データがここに反映されます。")
    st.stop()

df_log = pd.read_csv(LOG_PATH)
if df_log.empty:
    st.info("現在ログデータは空です。")
    st.stop()

for col in ['Actual', 'Result', 'Reason', 'Predicted']:
    if col not in df_log.columns:
        df_log[col] = None

df_log['Ticker_Name'] = df_log['Ticker'].map(get_company_name)
latest_date = df_log['Date'].max()

# 3. タブナビゲーション
tab1, tab2, tab3, tab4 = st.tabs(["本日の予測", "勝率・分析", "過去ログ", "銘柄管理"])

# --- TAB 1: 本日の予測 ---
with tab1:
    st.caption(f"最終更新: {latest_date}")
    df_latest = df_log[df_log['Date'] == latest_date].copy()
    
    up_count = (df_latest['Predicted'] == 1).sum()
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("本日シグナル対象", f"{len(df_latest)} 銘柄")
    with col_b:
        st.metric("上昇期待銘柄", f"{up_count} 銘柄")
        
    st.markdown("---")
    
    for _, row in df_latest.iterrows():
        is_up = row['Predicted'] == 1
        badge_html = '<span class="badge-up">上昇期待 (+0.5%以上)</span>' if is_up else '<span class="badge-down">静観 / 横ばい推移</span>'
        reason = row['Reason'] if pd.notna(row['Reason']) else '解析データなし'
        
        st.markdown(f"""
            <div class="stock-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #1a73e8;">
                        {row['Ticker_Name']} <span style="font-size: 0.8rem; color: #70757a; font-weight: normal;">({row['Ticker']})</span>
                    </h3>
                    {badge_html}
                </div>
                <div class="reason-box">
                    <strong>判断根拠:</strong><br>{reason}
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- TAB 2: 勝率・分析 ---
with tab2:
    st.subheader("モデル精度評価")
    df_eval = df_log.dropna(subset=['Result']).copy()
    
    if df_eval.empty:
        st.info("予測結果の答え合わせデータが集計されるまでお待ちください。")
    else:
        win_count = (df_eval['Result'] == "的中！").sum()
        total_count = len(df_eval)
        win_rate = (win_count / total_count) * 100 if total_count > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("総検証数", f"{total_count} 件")
        m2.metric("的中数", f"{win_count} 件")
        m3.metric("通算勝率", f"{win_rate:.1f} %")
        
        st.markdown("---")
        st.write("**銘柄別の勝率内訳**")
        
        ticker_stats = []
        for ticker in df_eval['Ticker'].unique():
            name = get_company_name(ticker)
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
    st.subheader("予測・結果データ一覧")
    
    selected_ticker = st.selectbox(
        "銘柄絞り込み", 
        ["すべての銘柄"] + list(df_log['Ticker_Name'].unique())
    )
    
    df_display = df_log.copy()
    if selected_ticker != "すべての銘柄":
        df_display = df_display[df_display['Ticker_Name'] == selected_ticker]
        
    df_display['予測'] = df_display['Predicted'].map({1: "上昇", 0: "静観"})
    
    target_cols = ['Date', 'Ticker_Name', '予測', 'Actual', 'Result', 'Reason']
    valid_cols = [c for c in target_cols if c in df_display.columns]
    
    df_table = df_display[valid_cols].sort_values(by="Date", ascending=False)
    
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

# --- TAB 4: 銘柄管理 ---
with tab4:
    st.subheader("監視銘柄の設定")
    st.caption("キーワード入力で候補が表示されます。追加した銘柄は次回の自動予測から反映されます。")
    
    current_tickers = load_tickers()
    
    st.write("**新規銘柄の検索・追加**")
    search_kw = st.text_input("銘柄名またはコードを入力 (例: トヨタ, 7203, 任天堂)", key="search_kw")
    
    if search_kw:
        candidates = search_stock_candidates(search_kw)
        if candidates:
            options = [f"{name} ({code})" for code, name in candidates.items()]
            selected_option = st.selectbox("候補から選択してください", options)
            
            if st.button("選択した銘柄を追加"):
                selected_code = selected_option.split("(")[-1].replace(")", "").strip()
                selected_name = candidates[selected_code]
                
                current_tickers[selected_code] = selected_name
                save_tickers(current_tickers)
                st.success(f"追加しました: {selected_name} ({selected_code})")
                st.rerun()
        else:
            st.warning("該当する銘柄が見つかりませんでした。4桁の銘柄コード（例: 7203）でお試しください。")
            
    st.markdown("---")
    
    st.write("**現在の監視銘柄一覧**")
    tickers_to_delete = []
    
    for code, name in current_tickers.items():
        col_info, col_del = st.columns([3, 1])
        with col_info:
            st.write(f"・ **{name}** ({code})")
        with col_del:
            if st.button("削除", key=f"del_{code}"):
                tickers_to_delete.append(code)
                
    if tickers_to_delete:
        for code in tickers_to_delete:
            del current_tickers[code]
        save_tickers(current_tickers)
        st.success("指定された銘柄を削除しました。")
        st.rerun()
