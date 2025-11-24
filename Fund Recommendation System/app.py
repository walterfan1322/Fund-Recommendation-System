import streamlit as st
import pandas as pd
import plotly.express as px
from fund_analyzer import FundAnalyzer
import os
import certifi
import shutil
import requests

# --- FIX FOR SSL ERROR ON WINDOWS WITH NON-ASCII PATHS ---
# yfinance's underlying library (curl_cffi) fails if the cert path has Chinese characters.
# We copy the cert to a safe location and point the environment variable there.
try:
    current_cert = certifi.where()
    safe_cert_path = os.path.join(os.path.expanduser("~"), ".gemini", "cacert.pem")
    os.makedirs(os.path.dirname(safe_cert_path), exist_ok=True)
    if not os.path.exists(safe_cert_path):
        shutil.copy(current_cert, safe_cert_path)
    os.environ['CURL_CA_BUNDLE'] = safe_cert_path
except Exception as e:
    print(f"Warning: Could not apply SSL fix: {e}")
# ---------------------------------------------------------

# --- Search Functionality ---
@st.cache_data(ttl=3600)
def search_yahoo(query):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if 'quotes' in data:
            return [f"{item['symbol']} - {item.get('shortname', item.get('longname', 'Unknown'))}" for item in data['quotes'] if 'symbol' in item]
        return []
    except Exception as e:
        st.error(f"搜尋失敗: {e}")
        return []

# Page Configuration
st.set_page_config(
    page_title="基金推薦系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Minimalist Design
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        font-family: 'Microsoft JhengHei', 'Helvetica Neue', sans-serif;
        color: #2c3e50;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ 設定")
    
    # --- Asset Type Selection ---
    asset_type = st.radio("資產類型", ["📊 ETF / 股票", "🏦 共同基金 (Mutual Funds)"])
    
    # --- Fund Data with Chinese Names ---
    # Format: "Ticker": "Display Name"
    
    etf_data = {
        "🇺🇸 美股大盤 (Broad Market)": {
            "SPY": "SPY - SPDR 標普500指數 ETF",
            "VOO": "VOO - Vanguard 標普500 ETF",
            "IVV": "IVV - iShares 核心標普500 ETF",
            "QQQ": "QQQ - Invesco 那斯達克100 ETF",
            "VTI": "VTI - Vanguard 整體股市 ETF",
            "DIA": "DIA - SPDR 道瓊工業指數 ETF",
            "IWM": "IWM - iShares 羅素2000 (小型股) ETF",
            "VEU": "VEU - Vanguard 全球(不含美國) ETF",
            "VTV": "VTV - Vanguard 價值股 ETF",
            "VUG": "VUG - Vanguard 成長股 ETF"
        },
        "💻 科技 (Technology)": {
            "XLK": "XLK - SPDR 科技類股 ETF",
            "VGT": "VGT - Vanguard 資訊科技 ETF",
            "SMH": "SMH - VanEck 半導體 ETF",
            "SOXX": "SOXX - iShares 半導體 ETF",
            "NVDA": "NVDA - 輝達 (NVIDIA)",
            "AAPL": "AAPL - 蘋果 (Apple)",
            "MSFT": "MSFT - 微軟 (Microsoft)",
            "TSLA": "TSLA - 特斯拉 (Tesla)",
            "AMD": "AMD - 超微半導體",
            "AVGO": "AVGO - 博通 (Broadcom)",
            "ARKK": "ARKK - ARK 創新主動型 ETF"
        },
        "💊 生技/醫療 (Healthcare)": {
            "XLV": "XLV - SPDR 醫療保健類股 ETF",
            "VHT": "VHT - Vanguard 醫療保健 ETF",
            "IBB": "IBB - iShares 那斯達克生技 ETF",
            "XBI": "XBI - SPDR 標普生技 ETF",
            "LLY": "LLY - 禮來藥廠 (Eli Lilly)",
            "UNH": "UNH - 聯合健康集團",
            "JNJ": "JNJ - 嬌生 (Johnson & Johnson)",
            "PFE": "PFE - 輝瑞 (Pfizer)"
        },
        "💰 金融 (Financials)": {
            "XLF": "XLF - SPDR 金融類股 ETF",
            "VFH": "VFH - Vanguard 金融 ETF",
            "JPM": "JPM - 摩根大通",
            "BAC": "BAC - 美國銀行",
            "V": "V - Visa",
            "MA": "MA - Mastercard",
            "BRK-B": "BRK-B - 波克夏海瑟威 B股"
        },
        "⚡ 能源/公用事業 (Energy/Utilities)": {
            "XLE": "XLE - SPDR 能源類股 ETF",
            "VDE": "VDE - Vanguard 能源 ETF",
            "XLU": "XLU - SPDR 公用事業類股 ETF",
            "XOM": "XOM - 艾克森美孚",
            "CVX": "CVX - 雪佛龍"
        },
        "🏠 不動產 (Real Estate)": {
            "VNQ": "VNQ - Vanguard 房地產 ETF",
            "XLRE": "XLRE - SPDR 房地產類股 ETF",
            "O": "O - Realty Income (月配息)",
            "AMT": "AMT - 美國電塔"
        },
        "🛡️ 債券 (Bonds)": {
            "BND": "BND - Vanguard 總體債券市場 ETF",
            "AGG": "AGG - iShares 核心美國總體債券 ETF",
            "TLT": "TLT - iShares 20年期以上美國公債 ETF",
            "IEF": "IEF - iShares 7-10年期美國公債 ETF",
            "SHV": "SHV - iShares 短期公債 ETF (現金管理)",
            "LQD": "LQD - iShares 投資等級公司債 ETF",
            "HYG": "HYG - iShares 高收益債 (垃圾債) ETF"
        },
        "🥇 黃金/原物料 (Commodities)": {
            "GLD": "GLD - SPDR 黃金 ETF",
            "IAU": "IAU - iShares 黃金信託 ETF",
            "SLV": "SLV - iShares 白銀 ETF",
            "DBC": "DBC - Invesco 德銀商品指數 ETF",
            "USO": "USO - 美國石油基金"
        }
    }

    mutual_fund_data = {
        "🤖 科技/AI (Tech/AI)": {
            "PGNAX": "PGNAX - 安聯 AI 人工智慧基金 (A股)",
            "FSELX": "FSELX - 富達半導體投資組合",
            "FSPTX": "FSPTX - 富達精選科技投資組合",
            "PRGTX": "PRGTX - T. Rowe Price 全球科技基金",
            "WSTAX": "WSTAX - Ivy 科學與技術基金"
        },
        "📈 成長型 (Growth)": {
            "VIGAX": "VIGAX - Vanguard 成長指數基金",
            "FBGRX": "FBGRX - 富達藍籌成長基金",
            "VWUSX": "VWUSX - Vanguard 美國成長基金",
            "AGTHX": "AGTHX - 美國成長基金 (American Funds)"
        },
        "🏢 平衡型 (Balanced)": {
            "VBIAX": "VBIAX - Vanguard 平衡指數基金 (60/40)",
            "VWELX": "VWELX - Vanguard 威靈頓基金",
            "FPURX": "FPURX - 富達清教徒基金"
        },
        "🌐 全球型 (Global)": {
            "VWIGX": "VWIGX - Vanguard 國際成長基金",
            "ODMAX": "ODMAX - 景順開發中市場基金",
            "ANWPX": "ANWPX - 新觀點基金 (American Funds)"
        },
        "🛡️ 債券型 (Bond)": {
            "VFIDX": "VFIDX - Vanguard 中期投資等級債券",
            "VBTLX": "VBTLX - Vanguard 總體債券市場指數",
            "PIMIX": "PIMIX - PIMCO 收益基金"
        }
    }

    # Select Data Source based on Asset Type
    if asset_type == "📊 ETF / 股票":
        current_data = etf_data
        st.caption("包含熱門美股 ETF 與個股。")
    else:
        current_data = mutual_fund_data
        st.caption("註: 共同基金代碼以美股代號為主 (例如 PGNAX 為安聯 AI 人工智慧基金 A 股)。")
    
    # 1. Category Selection
    st.subheader("1. 快速選擇")
    selected_category_name = st.selectbox("選擇分類", list(current_data.keys()))
    
    # Get options for the selected category
    # Create a list of "Ticker - Name" strings
    category_options = [f"{ticker} - {name.split(' - ')[-1]}" for ticker, name in current_data[selected_category_name].items()]
    
    # 2. Ticker Selection within Category
    selected_options = st.multiselect(
        f"選擇 {selected_category_name} 中的標的",
        category_options,
        default=category_options[:3] if len(category_options) >= 3 else category_options
    )
    
    st.markdown("---")
    
    # 3. Manual Input
    st.subheader("2. 手動輸入 / 補充")
    manual_tickers_input = st.text_area("輸入其他代碼 (用逗號分隔)", height=68, placeholder="例如: TSLA, AMD")
    
    st.markdown("---")
    st.subheader("🔍 搜尋標的")
    search_query = st.text_input("輸入關鍵字搜尋 (例如: Apple, 0050, 高股息)", placeholder="輸入後按 Enter")
    
    search_results = []
    if search_query:
        search_results = search_yahoo(search_query)
        
    selected_search_items = st.multiselect("搜尋結果", search_results)

    # Combine Tickers
    # Extract tickers from selected options (Format: "TICKER - Name")
    selected_tickers = [opt.split(' - ')[0] for opt in selected_options]
    searched_tickers = [opt.split(' - ')[0] for opt in selected_search_items]
    manual_tickers = [t.strip().upper() for t in manual_tickers_input.split(',') if t.strip()]
    all_tickers = list(set(selected_tickers + searched_tickers + manual_tickers))
    
    st.markdown("---")
    
    period_options = {
        "1y": "1 年",
        "3y": "3 年",
        "5y": "5 年",
        "10y": "10 年",
        "max": "最大範圍"
    }
    period = st.selectbox("回測期間", options=list(period_options.keys()), format_func=lambda x: period_options[x], index=2)
    
    analyze_btn = st.button("開始分析", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 關於")
    st.info("本工具分析歷史基金績效，協助您根據風險與報酬做出明智的決定。")

# Main Content
st.title("📈 基金推薦系統")
st.markdown("比較績效、波動率和風險調整後報酬，找出最適合您投資組合的基金。")

if analyze_btn:
    with st.spinner('正在抓取數據並進行運算...'):
        # Use the combined list
        tickers = all_tickers
        
        if not tickers:
            st.error("請至少選擇或輸入一個代碼。")
        else:
            # Initialize Analyzer
            analyzer = FundAnalyzer()
            
            # Fetch Data
            # Using st.cache_data to cache the result of this function
            @st.cache_data(ttl=3600)
            def get_data(t, p):
                return analyzer.fetch_data(t, p)
            
            try:
                raw_data = get_data(tickers, period)
                
                if raw_data.empty:
                    st.error(f"找不到數據。請檢查您的代碼是否正確，或網路連線是否正常。\n嘗試抓取的代碼: {tickers}")
                else:
                    # Calculate Metrics
                    metrics_df = analyzer.calculate_metrics(raw_data)
                    normalized_prices = analyzer.get_normalized_prices(raw_data)
                    
                    # --- Display Results ---
                    
                    # 1. Top Recommendations (Based on Sharpe Ratio)
                    st.subheader("🏆 最佳推薦")
                    
                    if not metrics_df.empty:
                        best_fund = metrics_df['Sharpe Ratio'].idxmax()
                        best_sharpe = metrics_df.loc[best_fund, 'Sharpe Ratio']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>最佳風險調整回報</h3>
                                <h1 style="color: #27ae60;">{best_fund}</h1>
                                <p>夏普比率 (Sharpe): {best_sharpe:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            highest_return_fund = metrics_df['CAGR'].idxmax()
                            highest_return = metrics_df.loc[highest_return_fund, 'CAGR']
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>最高報酬</h3>
                                <h1 style="color: #2980b9;">{highest_return_fund}</h1>
                                <p>年化報酬率: {highest_return:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col3:
                            lowest_risk_fund = metrics_df['Volatility'].idxmin()
                            lowest_risk = metrics_df.loc[lowest_risk_fund, 'Volatility']
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>最低風險</h3>
                                <h1 style="color: #8e44ad;">{lowest_risk_fund}</h1>
                                <p>波動率: {lowest_risk:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # 2. Metrics Table
                        st.markdown("### 📊 詳細指標")
                        
                        # Formatting for display
                        display_df = metrics_df.copy()
                        display_df.columns = ['年化報酬率 (CAGR)', '波動率 (Volatility)', '夏普比率 (Sharpe)', '最大回撤 (Max Drawdown)', '總報酬率 (Total Return)']
                        
                        display_df['年化報酬率 (CAGR)'] = display_df['年化報酬率 (CAGR)'].map('{:.2%}'.format)
                        display_df['波動率 (Volatility)'] = display_df['波動率 (Volatility)'].map('{:.2%}'.format)
                        display_df['最大回撤 (Max Drawdown)'] = display_df['最大回撤 (Max Drawdown)'].map('{:.2%}'.format)
                        display_df['夏普比率 (Sharpe)'] = display_df['夏普比率 (Sharpe)'].map('{:.2f}'.format)
                        display_df['總報酬率 (Total Return)'] = display_df['總報酬率 (Total Return)'].map('{:.2%}'.format)
                        
                        st.dataframe(display_df.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)
                        
                        # 3. Charts
                        st.markdown("### 📈 績效比較")
                        
                        # Line Chart
                        if not normalized_prices.empty:
                            fig_line = px.line(normalized_prices, x=normalized_prices.index, y=normalized_prices.columns, 
                                              title="累積報酬率 (歸一化為 100)",
                                              labels={"value": "價值 ($)", "variable": "代碼"})
                            fig_line.update_layout(hovermode="x unified", template="plotly_white")
                            st.plotly_chart(fig_line, use_container_width=True)
                        
                        # Scatter Plot (Risk vs Return)
                        st.markdown("### ⚖️ 風險 vs. 報酬分析")
                        fig_scatter = px.scatter(metrics_df.reset_index(), x="Volatility", y="CAGR", 
                                                text="Ticker", size=[10]*len(metrics_df),
                                                title="風險 (波動率) vs. 報酬 (年化)",
                                                labels={"CAGR": "年化報酬率", "Volatility": "年化波動率"},
                                                color="Sharpe Ratio", color_continuous_scale="Viridis")
                        fig_scatter.update_traces(textposition='top center')
                        fig_scatter.update_layout(template="plotly_white")
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                    else:
                        st.warning("無法計算指標，請檢查數據完整性。")
            except Exception as e:
                st.error(f"發生錯誤: {str(e)}")

else:
    st.info("👈 請在左側點擊「開始分析」按鈕。")
