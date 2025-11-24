import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict

class RecSysVisualizer:
    def __init__(self):
        # Set style if available, else default
        try:
            plt.style.use('seaborn-v0_8')
        except:
            pass

    def generate_explanation(self, user_profile: Dict, item_metrics: Dict) -> List[str]:
        """
        Generates natural language explanations for a recommendation.
        
        Args:
            user_profile: Dict with keys like 'risk_preference', 'investment_goal'.
            item_metrics: Dict with keys like 'sharpe_ratio', 'cagr', 'volatility', 'sector', 'risk_rating'.
            
        Returns:
            List of explanation strings.
        """
        reasons = []
        
        # 1. Performance-based
        if item_metrics.get('sharpe_ratio', 0) > 1.5:
            reasons.append(f"Excellent risk-adjusted return (Sharpe: {item_metrics['sharpe_ratio']:.2f}).")
        
        if item_metrics.get('cagr', 0) > 0.15: # 15%
            reasons.append(f"High growth potential with {item_metrics['cagr']*100:.1f}% annual return.")
            
        if item_metrics.get('volatility', 1.0) < 0.1: # 10%
            reasons.append("Low volatility option suitable for stable growth.")
            
        # 2. User Profile Match
        user_risk = user_profile.get('risk_preference', 'moderate')
        item_risk = item_metrics.get('risk_rating', 'moderate')
        
        # Simple mapping for demo
        if user_risk == item_risk:
            reasons.append(f"Matches your '{user_risk}' risk preference.")
            
        # 3. Sector/Interest (Mock logic)
        # In real app, check if item.sector is in user.interests
        
        if not reasons:
            reasons.append("Based on your overall investment profile.")
            
        return reasons

    def plot_risk_return(self, items_metrics: List[Dict], highlight_ids: List[int] = None):
        """
        Scatter plot of Risk (Volatility) vs Return (CAGR).
        """
        df = pd.DataFrame(items_metrics)
        
        plt.figure(figsize=(10, 6))
        
        # Plot all items
        plt.scatter(df['volatility'], df['cagr'], alpha=0.5, color='gray', label='Market')
        
        # Highlight recommended
        if highlight_ids:
            recommended = df[df['id'].isin(highlight_ids)]
            plt.scatter(recommended['volatility'], recommended['cagr'], color='red', s=100, label='Recommended')
            
            for _, row in recommended.iterrows():
                plt.annotate(f"Item {int(row['id'])}", (row['volatility'], row['cagr']), 
                             xytext=(5, 5), textcoords='offset points')
        
        plt.title('Risk vs. Return Analysis')
        plt.xlabel('Annualized Volatility (Risk)')
        plt.ylabel('CAGR (Return)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # In a real app, we might return the figure object or save it
        # plt.show() 
        return plt.gcf()

    def plot_price_history(self, price_sequences: Dict[int, np.ndarray], item_ids: List[int]):
        """
        Line chart of price history for selected items.
        """
        plt.figure(figsize=(10, 6))
        
        for iid in item_ids:
            if iid in price_sequences:
                prices = price_sequences[iid]
                # Normalize to start at 100
                normalized = (prices / prices[0]) * 100
                plt.plot(normalized, label=f"Item {iid}")
                
        plt.title('Historical Performance (Normalized)')
        plt.xlabel('Days')
        plt.ylabel('Value ($100 invested)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()

# --- Verification ---
if __name__ == "__main__":
    # Mock Data
    viz = RecSysVisualizer()
    
    user_profile = {'risk_preference': 'aggressive'}
    
    # Mock Metrics for 20 items
    items_metrics = []
    for i in range(20):
        items_metrics.append({
            'id': i,
            'sharpe_ratio': np.random.uniform(0.5, 2.5),
            'cagr': np.random.uniform(0.05, 0.30),
            'volatility': np.random.uniform(0.1, 0.4),
            'risk_rating': np.random.choice(['conservative', 'moderate', 'aggressive'])
        })
        
    # Generate Explanation for Item 0
    print("Explanation for Item 0:")
    reasons = viz.generate_explanation(user_profile, items_metrics[0])
    for r in reasons:
        print(f"- {r}")
        
    # Plot Risk-Return
    print("\nGenerating Risk-Return Plot...")
    fig1 = viz.plot_risk_return(items_metrics, highlight_ids=[0, 1, 2])
    fig1.savefig('risk_return_plot.png')
    print("Saved risk_return_plot.png")
    
    # Plot History
    print("\nGenerating History Plot...")
    price_seqs = {
        0: np.cumprod(1 + np.random.normal(0.001, 0.02, 100)) * 100,
        1: np.cumprod(1 + np.random.normal(0.001, 0.015, 100)) * 100,
        2: np.cumprod(1 + np.random.normal(0.0005, 0.01, 100)) * 100
    }
    fig2 = viz.plot_price_history(price_seqs, item_ids=[0, 1, 2])
    fig2.savefig('history_plot.png')
    print("Saved history_plot.png")
