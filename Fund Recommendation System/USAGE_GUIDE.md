# 推薦系統使用指南

## 🚀 快速開始

### 方式一：使用基金分析 Web UI（現有功能）

這是最簡單的使用方式，提供圖形介面來分析基金績效：

```bash
# 啟動 Streamlit 應用
streamlit run app.py
```

**功能**：
- ✅ 搜尋任何股票/ETF/基金（Yahoo Finance 資料）
- ✅ 視覺化績效比較（年化報酬、波動率、夏普比率）
- ✅ 風險 vs. 報酬散佈圖
- ✅ 最佳推薦（基於歷史數據）

### 方式二：使用推薦系統 API（新功能）

適合需要程式化調用或整合的場景。

## 📝 推薦系統完整範例

創建一個新檔案 `demo_recommendation.py`：

```python
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from recsys_data import DataSchema
from recsys_model import DualTowerRecModel
from recsys_train import TripletRecDataset, train_model
from recsys_inference import RecSysInference, ItemDataset
from recsys_viz import RecSysVisualizer

# ========== 步驟 1: 準備資料 ==========
print("步驟 1: 準備資料...")

# 用戶資料
users_df = pd.DataFrame({
    'user_id': [0, 1, 2],
    'age': [25, 40, 60],
    'risk_preference': ['aggressive', 'moderate', 'conservative'],
    'investment_goal': ['growth', 'retirement', 'savings'],
    'experience': ['novice', 'expert', 'average'],
    'income': [50000, 120000, 80000]
})

# 股票/基金資料（模擬 100 支）
num_items = 100
prices = np.random.rand(num_items, 50, 1) * 100

# 用戶互動紀錄（點擊/購買）
interactions = pd.DataFrame({
    'user_id': [0, 0, 1, 1, 2, 2],
    'item_id': [0, 5, 10, 15, 20, 25]
})

# 前處理
u_feats = DataSchema.preprocess_user_features(users_df)
i_seq = DataSchema.preprocess_item_sequence(prices, window_size=30)
i_static = {'category': torch.randint(0, 2, (num_items, 1))}

# ========== 步驟 2: 訓練模型 ==========
print("\n步驟 2: 訓練推薦模型...")

# 建立訓練資料集
dataset = TripletRecDataset(
    u_feats, i_static, i_seq, 
    interactions, 
    all_item_ids=np.arange(num_items)
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

# 設定模型
user_config = {
    'embedding_dims': {'risk_preference': (3, 8), 'investment_goal': (3, 8)},
    'numerical_dim': 2,
    'hidden_dims': [32],
    'output_dim': 16
}
item_config = {
    'embedding_dims': {'category': (2, 8)},
    'sequence_input_dim': 1,
    'hidden_dims': [32],
    'output_dim': 16
}

model = DualTowerRecModel(user_config, item_config)

# 訓練（5 個 epoch 示範）
train_model(model, dataloader, epochs=5, lr=1e-3, patience=3)

# ========== 步驟 3: 建立推薦引擎 ==========
print("\n步驟 3: 建立推薦引擎...")

# 準備所有商品的資料
item_dataset = ItemDataset(num_items=num_items)
item_dataset.static = i_static
item_dataset.sequence = i_seq
item_loader = DataLoader(item_dataset, batch_size=20)

# 初始化推論引擎
inference_engine = RecSysInference(model, item_loader)

# ========== 步驟 4: 為用戶生成推薦 ==========
print("\n步驟 4: 為用戶生成推薦...")

# 新用戶特徵
new_user = {
    'categorical': {
        'risk_preference': torch.tensor([[2]]),  # aggressive
        'investment_goal': torch.tensor([[0]])   # growth
    },
    'numerical': torch.tensor([[0.3, 0.6]])  # 正規化的 age, income
}

# 獲取 Top-5 推薦
recommendations = inference_engine.get_recommendations(new_user, k=5)

print("\n📊 推薦結果:")
for rank, (item_id, score) in enumerate(recommendations, 1):
    print(f"{rank}. Item {item_id} - 分數: {score:.4f}")

# ========== 步驟 5: 生成解釋與視覺化 ==========
print("\n步驟 5: 生成推薦解釋...")

viz = RecSysVisualizer()

# 模擬資產指標
item_metrics = {
    'id': recommendations[0][0],
    'sharpe_ratio': 2.1,
    'cagr': 0.18,
    'volatility': 0.12,
    'risk_rating': 'aggressive'
}

reasons = viz.generate_explanation(
    user_profile={'risk_preference': 'aggressive'},
    item_metrics=item_metrics
)

print(f"\n推薦理由 (Item {item_metrics['id']}):")
for r in reasons:
    print(f"  • {r}")

print("\n✅ 推薦系統演示完成！")
```

## 🎯 使用場景

### 場景 1: 快速分析現有基金
```bash
streamlit run app.py
# 在介面中搜尋 "AAPL" 或 "0050.TW"
```

### 場景 2: 訓練個人化推薦模型
```python
# 使用你的真實用戶資料和互動紀錄
python demo_recommendation.py
```

### 場景 3: 整合到現有系統
```python
from recsys_inference import RecSysInference

# 載入訓練好的模型
model = torch.load('trained_model.pth')
engine = RecSysInference(model, item_loader)

# API 調用
recs = engine.get_recommendations(user_features, k=10)
```

## 🔧 進階設定

### 自訂用戶特徵
```python
# 修改 recsys_data.py 中的映射
risk_map = {'保守': 0, '穩健': 1, '積極': 2, '激進': 3}
```

### 調整模型參數
```python
# 更大的嵌入維度
user_config = {
    'embedding_dims': {'risk_preference': (3, 16)},  # 8 -> 16
    'hidden_dims': [64, 32],  # 增加層數
    'output_dim': 32  # 16 -> 32
}
```

### 使用 FAISS 加速
```bash
pip install faiss-cpu
# 重新運行推論，會自動偵測並使用 FAISS
```

## 📊 視覺化範例

```python
from recsys_viz import RecSysVisualizer

viz = RecSysVisualizer()

# 繪製風險-報酬圖
items_metrics = [...]  # 你的資產指標列表
fig = viz.plot_risk_return(items_metrics, highlight_ids=[0, 1, 2])
fig.savefig('my_risk_return.png')

# 繪製歷史走勢
price_data = {...}  # {item_id: price_array}
fig = viz.plot_price_history(price_data, item_ids=[0, 1, 2])
fig.savefig('my_history.png')
```

## ❓ 常見問題

**Q: 我需要準備什麼資料？**  
A: 最少需要：
- 用戶特徵（年齡、風險偏好等）
- 資產歷史價格/淨值
- 用戶-資產互動紀錄（點擊/購買）

**Q: 可以預測未來收益嗎？**  
A: 不行，這是推薦系統，基於歷史偏好匹配，不是預測模型。

**Q: 如何整合真實的 Yahoo Finance 資料？**  
A: 使用 `yfinance` 抓取歷史價格，參考 `fund_analyzer.py` 的實作。

## 🎓 下一步

1. **收集真實資料**：用戶行為日誌、資產資料庫
2. **調優模型**：嘗試不同超參數、增加特徵
3. **部署上線**：整合到 Web 服務或 App
4. **持續學習**：定期用新資料重新訓練
