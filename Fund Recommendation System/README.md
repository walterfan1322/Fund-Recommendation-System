# 基金推薦系統 (Fund Recommendation System)

這是一個基於 Python 和 Streamlit 構建的基金與股票分析推薦系統。它利用 Yahoo Finance 的數據，幫助投資者比較不同資產（ETF、股票、共同基金）的績效、風險和夏普比率。

## ✨ 功能特色

-   **多資產支援**：支援美股、ETF、共同基金，以及透過搜尋功能支援全球市場標的（如台股 0050.TW）。
-   **即時搜尋**：內建 Yahoo Finance 搜尋功能，可輸入關鍵字（如 "Apple", "High Dividend"）查找任何標的。
-   **績效分析**：計算年化報酬率 (CAGR)、波動率 (Volatility)、夏普比率 (Sharpe Ratio) 和最大回撤 (Max Drawdown)。
-   **視覺化圖表**：提供累積報酬率走勢圖與風險/報酬散佈圖。
-   **最佳推薦**：自動標示出「最佳風險調整回報」、「最高報酬」與「最低風險」的標的。

## 🛠️ 安裝與執行

### 1. 環境設定

請確保您已安裝 Python 3.8 或以上版本。

建議使用虛擬環境：

```bash
# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境 (Windows)
.\.venv\Scripts\activate
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 執行程式

```bash
streamlit run app.py
```

程式啟動後，瀏覽器將自動開啟應用程式（預設為 `http://localhost:8501`）。

## 📂 專案結構

-   `app.py`: 主程式，包含 Streamlit 介面與邏輯。
-   `fund_analyzer.py`: 核心分析類別，負責抓取數據與計算指標。
-   `requirements.txt`: 專案依賴套件列表。
-   `.venv/`: 虛擬環境目錄 (安裝後產生)。

## 📝 注意事項

-   本系統數據來源為 Yahoo Finance，僅供參考，不構成投資建議。
-   部分共同基金代碼可能需要加上後綴（如台股需加 `.TW`，港股需加 `.HK`），建議使用內建搜尋功能確認代碼。
