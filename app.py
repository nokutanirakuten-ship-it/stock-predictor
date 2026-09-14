import os
import json
import base64
import requests
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# 1. ページ基本設定
st.set_page_config(
    page_title="AI株価予測 & 配当管理",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 720px;
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
    """旧フォーマット(辞書形式)にも対応したデータの読み込み"""
    if os.path.exists(TICKER_JSON_PATH):
        try:
            with open(TICKER_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                formatted = {}
                for k, v in data.items():
                    if isinstance(v, str):
                        formatted[k] = {"name": v, "shares": 0, "buy_price": 0.0}
                    else:
                        formatted[k] = v
                return formatted
        except Exception:
            pass
    return {
        "7203.T": {"name": "トヨタ自動車", "shares": 100, "buy_price": 2500.0},
        "6758.T": {"name": "ソニーグループ", "shares": 50, "buy_price": 12000.0}
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
    candidates = {}
    if not query:
        return candidates
    query_str = str(query).strip()
    
    if query_str.isdigit() and len(query_str) == 4:
        symbol = f"{query_str}.T"
        try:
            info = yf.Ticker(symbol).info
            name = info.get('shortName') or info.get('longName') or symbol
            candidates[symbol] = name
        except Exception:
            candidates[symbol] = f"銘柄コード {symbol}"
            
    try:
        search_res = yf.Search(query_str, max_results=8)
        quotes = getattr(search_res, 'quotes', [])
        for q in quotes:
            symbol = q.get('symbol', '')
            name = q.get('shortname') or q.get('longname') or symbol
            if symbol.endswith('.T'):
                candidates[symbol] = name
    except Exception:
        pass
    return candidates

TICKER_DATA = load_tickers()

@st.cache_data
def get_company_name(ticker):
    if ticker in TICKER_DATA:
        return TICKER_DATA[ticker]["name"]
    try:
        info = yf.Ticker(ticker).info
        return info.get('shortName') or info.get('longName') or ticker
    except Exception:
        return ticker

@st.cache_data
def fetch_dividend_history(tickers_dict):
    """過去の配当データと今後の月別入金予測データを算出"""
    dividend_records = []
    monthly_schedule = {m: 0.0 for m in range(1, 13)}
    
    for ticker, meta in tickers_dict.items():
        shares = meta.get("shares", 0)
        if shares <= 0:
            continue
            
        try:
            t = yf.Ticker(ticker)
            divs = t.dividends
            if divs.empty:
                continue
                
            # タイムゾーン除去
            divs.index = divs.index.tz_localize(None)
            
            # 1年以内の月別入金予定計算
            one_year_ago = datetime.now() - timedelta(days=365)
            recent_divs = divs[divs.index >= one_year_ago]
            
            for dt, val in recent_divs.items():
                pay_month = dt.month
                monthly_schedule[pay_month] += val * shares
                
            for dt, val in divs.items():
                dividend_records.append({
                    "Date": dt,
                    "Ticker": ticker,
                    "Name": meta["name"],
                    "Amount": val * shares
                })
        except Exception:
            continue
            
    df_divs = pd.DataFrame(dividend_records) if dividend_records else pd.DataFrame(columns=["Date", "Ticker", "Name", "Amount"])
    return df_divs, monthly_schedule

# 2. アプリヘッダー
st.title("AI株価予測 & 配当管理")
st.caption("LightGBM x SHAP AI予測 & ポートフォリオ配当分析")

df_log = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()

# 3. タブナビゲーション
tab1, tab2, tab3, tab4, tab5 = st.tabs(["本日の予測", "配当管理", "勝率・分析", "過去ログ", "銘柄・ポートフォリオ管理"])

# --- TAB 1: 本日の予測 ---
with tab1:
    if df_log.empty:
        st.info("予測データが見つかりません。自動バッチ処理の完了をお待ちください。")
    else:
        latest_date = df_log['Date'].max()
        st.caption(f"最終更新: {latest_date}")
        df_latest = df_log[df_log['Date'] == latest_date].copy()
        df_latest['Ticker_Name'] = df_latest['Ticker'].map(get_company_name)
        
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
                            {row['Ticker_Name']} <span style="font-size: 0.8rem; color: #70757a;">({row['Ticker']})</span>
                        </h3>
                        {badge_html}
                    </div>
                    <div class="reason-box"><strong>判断根拠:</strong><br>{reason}</div>
                </div>
            """, unsafe_allow_html=True)

# --- TAB 2: 配当管理 ---
with tab2:
    st.subheader("ポートフォリオ配当分析")
    
    df_divs, monthly_schedule = fetch_dividend_history(TICKER_DATA)
    
    # 全体サマリー計算
    total_asset = 0.0
    annual_div_est = sum(monthly_schedule.values())
    
    for code, meta in TICKER_DATA.items():
        total_asset += meta.get("shares", 0) * meta.get("buy_price", 0.0)
        
    div_yield_est = (annual_div_est / total_asset * 100) if total_asset > 0 else 0.0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("総投資額", f"¥{total_asset:,.0f}")
    m2.metric("年間予想配当金", f"¥{annual_div_est:,.0f}")
    m3.metric("予想配当利回り", f"{div_yield_est:.2f} %")
    
    st.markdown("---")
    
    # 1. 月別配当金グラフ (いつ振り込まれるか？)
    st.write("**月別配当金・入金スケジュール（年間推移）**")
    df_monthly = pd.DataFrame({
        "月": [f"{m}月" for m in range(1, 13)],
        "配当金(円)": [monthly_schedule[m] for m in range(1, 13)]
    })
    st.bar_chart(df_monthly.set_index("月"), height=250)
    
    st.markdown("---")
    
    # 2. 期間別（1W, 1M, 3M, 1Y）の配当推移グラフ
    st.write("**配当金の受け取り推移**")
    time_range = st.radio("表示期間を選択", ["1W", "1M", "3M", "1Y", "全期間"], horizontal=True, index=3)
    
    if not df_divs.empty:
        now = datetime.now()
        if time_range == "1W":
            start_date = now - timedelta(days=7)
        elif time_range == "1M":
            start_date = now - timedelta(days=30)
        elif time_range == "3M":
            start_date = now - timedelta(days=90)
        elif time_range == "1Y":
            start_date = now - timedelta(days=365)
        else:
            start_date = df_divs["Date"].min()
            
        df_filtered = df_divs[df_divs["Date"] >= start_date].sort_values(by="Date")
        
        if not df_filtered.empty:
            df_chart = df_filtered.groupby("Date")["Amount"].sum().reset_index()
            df_chart["累計配当額(円)"] = df_chart["Amount"].cumsum()
            st.line_chart(df_chart.set_index("Date")[["累計配当額(円)"]], height=260)
        else:
            st.info("選択された期間内に受け取った配当履歴はありません。")
    else:
        st.info("保有株数が設定されている銘柄がありません。「銘柄・ポートフォリオ管理」タブで株数を設定してください。")

# --- TAB 3: 勝率・分析 ---
with tab3:
    st.subheader("モデル精度評価")
    if not df_log.empty and 'Result' in df_log.columns:
        df_eval = df_log.dropna(subset=['Result']).copy()
        if df_eval.empty:
            st.info("予測結果の答え合わせデータが開示されるまでお待ちください。")
        else:
            win_count = (df_eval['Result'] == "的中！").sum()
            total_count = len(df_eval)
            win_rate = (win_count / total_count) * 100 if total_count > 0 else 0
            
            m1, m2, m3 = st.columns(3)
            m1.metric("総検証数", f"{total_count} 件")
            m2.metric("的中数", f"{win_count} 件")
            m3.metric("通算勝率", f"{win_rate:.1f} %")
    else:
        st.info("ログデータが存在しません。")

# --- TAB 4: 過去ログ ---
with tab4:
    st.subheader("予測・結果データ一覧")
    if not df_log.empty:
        df_log['Ticker_Name'] = df_log['Ticker'].map(get_company_name)
        selected_ticker = st.selectbox("銘柄絞り込み", ["すべての銘柄"] + list(df_log['Ticker_Name'].unique()))
        
        df_display = df_log.copy()
        if selected_ticker != "すべての銘柄":
            df_display = df_display[df_display['Ticker_Name'] == selected_ticker]
            
        df_display['予測'] = df_display['Predicted'].map({1: "上昇", 0: "静観"})
        target_cols = ['Date', 'Ticker_Name', '予測', 'Actual', 'Result', 'Reason']
        valid_cols = [c for c in target_cols if c in df_display.columns]
        
        st.dataframe(df_display[valid_cols].sort_values(by="Date", ascending=False), use_container_width=True, hide_index=True)

# --- TAB 5: 銘柄・ポートフォリオ管理 ---
with tab5:
    st.subheader("監視・保有銘柄の設定")
    st.caption("銘柄の検索・保有株数・取得単価を登録できます。設定は自動予測と配当管理に反映されます。")
    
    current_tickers = load_tickers()
    
    st.write("**新規銘柄の検索・追加**")
    search_kw = st.text_input("銘柄名またはコードを入力 (例: トヨタ, 7203, 任天堂)", key="search_kw")
    
    if search_kw:
        candidates = search_stock_candidates(search_kw)
        if candidates:
            options = [f"{name} ({code})" for code, name in candidates.items()]
            selected_option = st.selectbox("候補から選択してください", options)
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                add_shares = st.number_input("保有株数", min_value=0, value=100, step=10)
            with col_s2:
                add_price = st.number_input("平均取得単価 (円)", min_value=0.0, value=0.0, step=100.0)
                
            if st.button("選択した銘柄を登録"):
                selected_code = selected_option.split("(")[-1].replace(")", "").strip()
                selected_name = candidates[selected_code]
                
                current_tickers[selected_code] = {
                    "name": selected_name,
                    "shares": int(add_shares),
                    "buy_price": float(add_price)
                }
                save_tickers(current_tickers)
                st.success(f"登録しました: {selected_name} ({selected_code})")
                st.rerun()
        else:
            st.warning("該当する銘柄が見つかりませんでした。4桁のコードをお試しください。")
            
    st.markdown("---")
    
    st.write("**登録済み銘柄・保有資産一覧**")
    tickers_to_delete = []
    
    for code, meta in current_tickers.items():
        with st.expander(f"・ **{meta['name']}** ({code}) — 保有: {meta.get('shares', 0)} 株"):
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                new_shares = st.number_input("保有株数", min_value=0, value=int(meta.get("shares", 0)), key=f"s_{code}")
            with col_e2:
                new_price = st.number_input("平均取得単価", min_value=0.0, value=float(meta.get("buy_price", 0.0)), key=f"p_{code}")
                
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("更新", key=f"upd_{code}"):
                    current_tickers[code]["shares"] = int(new_shares)
                    current_tickers[code]["buy_price"] = float(new_price)
                    save_tickers(current_tickers)
                    st.success("更新しました。")
                    st.rerun()
            with col_btn2:
                if st.button("削除", key=f"del_{code}"):
                    tickers_to_delete.append(code)
                    
    if tickers_to_delete:
        for code in tickers_to_delete:
            del current_tickers[code]
        save_tickers(current_tickers)
        st.success("銘柄を削除しました。")
        st.rerun()
