import yfinance as yf
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Tuple

class RealDataLoader:
    def __init__(self):
        # Default candidate pool (Mix of US Stocks, ETFs, and TW Stocks)
        self.default_tickers = [
            "SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", # US ETFs
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", # US Tech
            "JPM", "BAC", "V", "MA", # US Finance
            "JNJ", "PFE", "UNH", # US Healthcare
            "0050.TW", "0056.TW", "2330.TW", "2317.TW", "2454.TW" # TW Stocks
        ]
        
    def fetch_data(self, tickers: List[str] = None, period="1y") -> Tuple[Dict, torch.Tensor, List[Dict]]:
        """
        Fetches data from Yahoo Finance and prepares it for the model.
        
        Returns:
            item_static: Dict of tensors
            item_sequence: Tensor (Num_Items, Seq_Len, Features)
            item_metrics: List of Dicts (for visualization)
        """
        if tickers is None:
            tickers = self.default_tickers
            
        print(f"Fetching data for {len(tickers)} assets...")
        
        # Download data
        # auto_adjust=True handles splits and dividends
        data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
        
        # Handle single ticker case where columns are not MultiIndex
        if len(tickers) == 1:
            # Reformat to match MultiIndex structure if needed, or just handle differently
            # For simplicity, let's assume we always have multiple tickers or handle the shape
            pass

        # Extract Close prices
        # Shape: (Dates, Tickers)
        try:
            close_prices = data['Close']
        except KeyError:
            # Fallback if 'Close' is not found (sometimes yf structure varies)
            close_prices = data
            
        # Ensure columns match tickers list order
        # Filter out failed downloads
        valid_tickers = [t for t in tickers if t in close_prices.columns]
        close_prices = close_prices[valid_tickers]
        
        # Fill missing values
        close_prices = close_prices.ffill().bfill()
        
        # --- 1. Prepare Sequence Data (for Model) ---
        # Normalize: Percentage change or MinMax?
        # Model expects (Num_Items, Seq_Len, 1)
        # Let's take last 30 days
        window_size = 30
        sequences = []
        
        for ticker in valid_tickers:
            series = close_prices[ticker].values
            if len(series) < window_size:
                # Pad if too short
                seq = np.pad(series, (window_size - len(series), 0), 'edge')
            else:
                seq = series[-window_size:]
                
            # Normalize: (Price / First_Price) - 1  (Cumulative Return relative to start of window)
            # Or Z-score? Let's use Z-score for stability in neural nets
            mean = seq.mean()
            std = seq.std() + 1e-8
            norm_seq = (seq - mean) / std
            
            sequences.append(norm_seq.reshape(-1, 1))
            
        item_sequence = torch.tensor(np.array(sequences), dtype=torch.float32)
        
        # --- 2. Prepare Static Data (for Model) ---
        # Mocking category for now since yf doesn't give sector easily in bulk download
        # In real app, we'd query Ticker.info one by one (slow) or use a static mapping
        # Let's assign random categories for demo: 0=Stock, 1=ETF
        categories = []
        for t in valid_tickers:
            if "00" in t or "SPY" in t or "QQQ" in t or "VTI" in t: # Simple heuristic
                categories.append(1) # ETF
            else:
                categories.append(0) # Stock
                
        item_static = {
            'category': torch.tensor(categories).unsqueeze(1)
        }
        
        # --- 3. Calculate Metrics (for Visualization) ---
        item_metrics = []
        for i, ticker in enumerate(valid_tickers):
            series = close_prices[ticker].values
            
            # Calculate Daily Returns
            returns = np.diff(series) / series[:-1]
            
            # CAGR (approx for 1 year)
            total_return = (series[-1] / series[0]) - 1
            
            # Volatility (Annualized)
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe (Assume Rf=2%)
            sharpe = (np.mean(returns) * 252 - 0.02) / (volatility + 1e-8)
            
            item_metrics.append({
                'id': i, # Internal ID matches index in tensor
                'ticker': ticker,
                'cagr': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'risk_rating': 'aggressive' if volatility > 0.2 else 'moderate' if volatility > 0.1 else 'conservative',
                'prices': series # Store full history for plotting
            })
            
        return item_static, item_sequence, item_metrics, valid_tickers

if __name__ == "__main__":
    loader = RealDataLoader()
    static, seq, metrics, tickers = loader.fetch_data()
    print(f"Loaded {len(tickers)} assets.")
    print("Sequence Shape:", seq.shape)
    print("Sample Metric:", metrics[0])
