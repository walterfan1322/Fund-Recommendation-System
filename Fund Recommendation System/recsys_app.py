import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import our modules
from data_loader import RealDataLoader
from recsys_data import DataSchema
from recsys_model import DualTowerRecModel
from recsys_train import TripletRecDataset, train_model
from recsys_inference import RecSysInference, ItemDataset
from recsys_viz import RecSysVisualizer

# Page Config
st.set_page_config(page_title="Fund RecSys AI", layout="wide")

# --- Session State Initialization ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'item_metrics' not in st.session_state:
    st.session_state.item_metrics = []
if 'valid_tickers' not in st.session_state:
    st.session_state.valid_tickers = []

# --- Sidebar: User Profile ---
st.sidebar.header("👤 用戶投資檔案")

age = st.sidebar.slider("年齡", 18, 80, 30)
income = st.sidebar.slider("年收入 (萬)", 30, 500, 80)
risk_pref = st.sidebar.selectbox("風險偏好", ["保守 (Conservative)", "穩健 (Moderate)", "積極 (Aggressive)"])
goal = st.sidebar.selectbox("投資目標", ["資產增值 (Growth)", "退休規劃 (Retirement)", "儲蓄保本 (Savings)"])
experience = st.sidebar.selectbox("投資經驗", ["新手 (Novice)", "普通 (Average)", "專家 (Expert)"])

# Map inputs to model format
risk_map = {"保守 (Conservative)": "conservative", "穩健 (Moderate)": "moderate", "積極 (Aggressive)": "aggressive"}
goal_map = {"資產增值 (Growth)": "growth", "退休規劃 (Retirement)": "retirement", "儲蓄保本 (Savings)": "savings"}
exp_map = {"新手 (Novice)": "novice", "普通 (Average)": "average", "專家 (Expert)": "expert"}

user_profile = {
    'age': age,
    'income': income * 10000,
    'risk_preference': risk_map[risk_pref],
    'investment_goal': goal_map[goal],
    'experience': exp_map[experience]
}

# --- Main Area ---
st.title("🤖 股票與基金個人化推薦系統")
st.markdown("結合 **雙塔深度學習模型 (Dual-Tower)** 與 **Yahoo Finance 即時數據**")

# --- Step 1: Data Loading ---
st.header("1. 市場數據與偏好")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🔄 下載最新市場數據 (Yahoo Finance)"):
        with st.spinner("正在下載並處理數據..."):
            loader = RealDataLoader()
            # Fetch default list
            static, seq, metrics, tickers = loader.fetch_data()
            
            st.session_state.item_static = static
            st.session_state.item_sequence = seq
            st.session_state.item_metrics = metrics
            st.session_state.valid_tickers = tickers
            st.session_state.data_loaded = True
            st.success(f"成功載入 {len(tickers)} 支標的！")

if st.session_state.data_loaded:
    # User Interactions (Likes)
    liked_assets = st.multiselect(
        "❤️ 選擇您感興趣或持有的標的 (用於訓練模型偏好):",
        st.session_state.valid_tickers,
        default=st.session_state.valid_tickers[:3] # Default select a few
    )
    
    # --- Step 2: Model Training ---
    st.header("2. 訓練個人化模型")
    
    if st.button("🚀 開始訓練模型"):
        if not liked_assets:
            st.warning("請至少選擇一個感興趣的標的！")
        else:
            with st.spinner("正在訓練雙塔模型 (Dual-Tower)..."):
                # 1. Prepare Data
                # Create a single user DataFrame
                users_df = pd.DataFrame([user_profile])
                users_df['user_id'] = 0 # Single user mode
                
                u_feats = DataSchema.preprocess_user_features(users_df)
                
                # Create Interactions DataFrame
                # User 0 likes selected assets
                liked_indices = [st.session_state.valid_tickers.index(t) for t in liked_assets]
                interactions = pd.DataFrame({
                    'user_id': [0] * len(liked_indices),
                    'item_id': liked_indices
                })
                
                # Dataset
                dataset = TripletRecDataset(
                    u_feats, 
                    st.session_state.item_static, 
                    st.session_state.item_sequence, 
                    interactions, 
                    all_item_ids=np.arange(len(st.session_state.valid_tickers))
                )
                # Need drop_last=False if we have few interactions, but TripletRecDataset samples negatives dynamically
                # Batch size must be smaller than interactions if possible, or just 1 if very few
                bs = min(4, len(interactions))
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
                
                # Model Config
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
                
                # Train
                train_model(model, dataloader, epochs=10, lr=0.005) # Higher LR for fast demo
                
                st.session_state.model = model
                st.success("模型訓練完成！")

    # --- Step 3: Recommendations ---
    if st.session_state.model is not None:
        st.header("3. 您的專屬推薦")
        
        # Inference
        # Prepare Item Loader for Inference
        class SimpleItemDataset(ItemDataset):
            def __init__(self, static, sequence, num_items):
                super().__init__(num_items)
                self.static = static
                self.sequence = sequence
                
        item_ds = SimpleItemDataset(
            st.session_state.item_static, 
            st.session_state.item_sequence, 
            len(st.session_state.valid_tickers)
        )
        item_loader = DataLoader(item_ds, batch_size=20)
        
        engine = RecSysInference(st.session_state.model, item_loader)
        
        # Prepare User Features for Inference
        # Reuse u_feats from training prep (it's for user_id 0)
        # We need to extract the tensors for the single user
        users_df = pd.DataFrame([user_profile])
        users_df['user_id'] = 0
        u_feats_all = DataSchema.preprocess_user_features(users_df)
        
        user_input = {
            'categorical': {k: v[0].unsqueeze(0) for k, v in u_feats_all.items() if k in ['risk_preference', 'investment_goal']},
            'numerical': torch.cat([u_feats_all['age'], u_feats_all['income']], dim=1)[0].unsqueeze(0)
        }
        
        recs = engine.get_recommendations(user_input, k=5)
        
        # Display Recommendations
        viz = RecSysVisualizer()
        
        rec_cols = st.columns(5)
        for i, (item_id, score) in enumerate(recs):
            metric = st.session_state.item_metrics[item_id]
            ticker = metric['ticker']
            
            with rec_cols[i]:
                st.subheader(f"{i+1}. {ticker}")
                st.caption(f"匹配分: {score:.2f}")
                
                # Explanation
                reasons = viz.generate_explanation(user_profile, metric)
                for r in reasons[:2]: # Show top 2 reasons
                    st.write(f"• {r}")
                    
                st.metric("年化報酬", f"{metric['cagr']*100:.1f}%")
                st.metric("夏普比率", f"{metric['sharpe_ratio']:.2f}")

        # --- Visualizations ---
        st.subheader("📊 深度分析")
        
        tab1, tab2 = st.tabs(["風險-報酬分析", "歷史走勢比較"])
        
        rec_ids = [r[0] for r in recs]
        
        with tab1:
            fig1 = viz.plot_risk_return(st.session_state.item_metrics, highlight_ids=rec_ids)
            st.pyplot(fig1)
            
        with tab2:
            # Prepare price data dict
            price_data = {m['id']: m['prices'] for m in st.session_state.item_metrics if m['id'] in rec_ids}
            fig2 = viz.plot_price_history(price_data, item_ids=rec_ids)
            st.pyplot(fig2)

else:
    st.info("請先點擊上方按鈕下載市場數據。")
