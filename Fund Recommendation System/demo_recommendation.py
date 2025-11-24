"""
推薦系統完整演示
展示從訓練到推論的完整流程
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from recsys_data import DataSchema
from recsys_model import DualTowerRecModel
from recsys_train import TripletRecDataset, train_model
from recsys_inference import RecSysInference, ItemDataset
from recsys_viz import RecSysVisualizer

def main():
    print("=" * 60)
    print("基金與股票個人化推薦系統 - 完整演示")
    print("=" * 60)
    
    # ========== 步驟 1: 準備資料 ==========
    print("\n[1/5] 準備資料...")
    
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
        'user_id': [0, 0, 1, 1, 2, 2, 0, 1],
        'item_id': [0, 5, 10, 15, 20, 25, 30, 35]
    })
    
    print(f"   ✓ 用戶數: {len(users_df)}")
    print(f"   ✓ 資產數: {num_items}")
    print(f"   ✓ 互動數: {len(interactions)}")
    
    # 前處理
    u_feats = DataSchema.preprocess_user_features(users_df)
    i_seq = DataSchema.preprocess_item_sequence(prices, window_size=30)
    i_static = {'category': torch.randint(0, 2, (num_items, 1))}
    
    # ========== 步驟 2: 訓練模型 ==========
    print("\n[2/5] 訓練推薦模型...")
    
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
    print("\n[3/5] 建立推薦引擎...")
    
    # 準備所有商品的資料
    class SimpleItemDataset(ItemDataset):
        def __init__(self, static, sequence, num_items):
            super().__init__(num_items)
            self.static = static
            self.sequence = sequence
    
    item_dataset = SimpleItemDataset(i_static, i_seq, num_items)
    item_loader = DataLoader(item_dataset, batch_size=20)
    
    # 初始化推論引擎
    inference_engine = RecSysInference(model, item_loader)
    print("   ✓ 索引建立完成")
    
    # ========== 步驟 4: 為用戶生成推薦 ==========
    print("\n[4/5] 為用戶生成推薦...")
    
    # 三種不同類型的用戶
    test_users = [
        {
            'name': '積極型年輕投資人',
            'features': {
                'categorical': {
                    'risk_preference': torch.tensor([[2]]),  # aggressive
                    'investment_goal': torch.tensor([[0]])   # growth
                },
                'numerical': torch.tensor([[0.2, 0.4]])  # 年輕、中等收入
            }
        },
        {
            'name': '穩健型中年投資人',
            'features': {
                'categorical': {
                    'risk_preference': torch.tensor([[1]]),  # moderate
                    'investment_goal': torch.tensor([[1]])   # retirement
                },
                'numerical': torch.tensor([[0.6, 0.8]])  # 中年、高收入
            }
        },
        {
            'name': '保守型退休投資人',
            'features': {
                'categorical': {
                    'risk_preference': torch.tensor([[0]]),  # conservative
                    'investment_goal': torch.tensor([[2]])   # savings
                },
                'numerical': torch.tensor([[0.9, 0.6]])  # 年長、中高收入
            }
        }
    ]
    
    print("\n" + "=" * 60)
    for user in test_users:
        print(f"\n👤 {user['name']}:")
        recommendations = inference_engine.get_recommendations(user['features'], k=5)
        
        print("   推薦清單 (Top-5):")
        for rank, (item_id, score) in enumerate(recommendations, 1):
            print(f"      {rank}. Item {item_id:3d} - 匹配分數: {score:6.4f}")
    
    # ========== 步驟 5: 生成解釋與視覺化 ==========
    print("\n" + "=" * 60)
    print("\n[5/5] 生成推薦解釋與視覺化...")
    
    viz = RecSysVisualizer()
    
    # 為第一個推薦生成解釋
    test_user = test_users[0]
    top_item = recommendations[0][0]
    
    # 模擬資產指標
    item_metrics = {
        'id': top_item,
        'sharpe_ratio': np.random.uniform(1.5, 2.5),
        'cagr': np.random.uniform(0.12, 0.25),
        'volatility': np.random.uniform(0.10, 0.20),
        'risk_rating': 'aggressive'
    }
    
    reasons = viz.generate_explanation(
        user_profile={'risk_preference': 'aggressive'},
        item_metrics=item_metrics
    )
    
    print(f"\n📊 推薦理由 (Item {item_metrics['id']}):")
    for r in reasons:
        print(f"   • {r}")
    
    # 生成視覺化
    print("\n📈 生成視覺化圖表...")
    
    # 模擬所有資產的指標
    all_metrics = []
    for i in range(num_items):
        all_metrics.append({
            'id': i,
            'sharpe_ratio': np.random.uniform(0.5, 2.5),
            'cagr': np.random.uniform(0.05, 0.30),
            'volatility': np.random.uniform(0.1, 0.4),
        })
    
    # 繪製風險-報酬圖
    highlight_ids = [rec[0] for rec in recommendations[:5]]
    fig1 = viz.plot_risk_return(all_metrics, highlight_ids=highlight_ids)
    fig1.savefig('demo_risk_return.png')
    print("   ✓ 已儲存: demo_risk_return.png")
    
    # 繪製歷史走勢
    price_data = {}
    for item_id in highlight_ids[:3]:
        price_data[item_id] = np.cumprod(1 + np.random.normal(0.001, 0.02, 100)) * 100
    
    fig2 = viz.plot_price_history(price_data, item_ids=list(price_data.keys()))
    fig2.savefig('demo_history.png')
    print("   ✓ 已儲存: demo_history.png")
    
    # 完成
    print("\n" + "=" * 60)
    print("✅ 推薦系統演示完成！")
    print("=" * 60)
    print("\n下一步：")
    print("  1. 查看生成的圖表: demo_risk_return.png, demo_history.png")
    print("  2. 使用真實資料訓練模型")
    print("  3. 整合到你的應用中")

if __name__ == "__main__":
    main()
