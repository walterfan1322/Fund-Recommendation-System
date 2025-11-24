import yfinance as yf
import pandas as pd
import numpy as np

class FundAnalyzer:
    def __init__(self):
        pass

    def fetch_data(self, tickers, period="5y"):
        """
        Fetches historical data for the given tickers.
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Download data
        # auto_adjust=True gets the adjusted close directly
        try:
            data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, threads=True)
        except Exception as e:
            print(f"Download failed: {e}")
            return pd.DataFrame()
        
        if data.empty:
            return data

        # If only one ticker, yfinance returns a DataFrame with columns like 'Open', 'Close' etc.
        # We want to force it to be a MultiIndex with levels (Ticker, OHLCV) to be consistent
        if len(tickers) == 1:
            ticker = tickers[0]
            # Check if it's already MultiIndex (sometimes yfinance does this unpredictably)
            if isinstance(data.columns, pd.MultiIndex):
                pass 
            else:
                # Reconstruct columns
                data.columns = pd.MultiIndex.from_product([[ticker], data.columns])
            
        return data

    def calculate_metrics(self, data, risk_free_rate=0.02):
        """
        Calculates performance and risk metrics.
        """
        metrics = []
        
        # Extract Close prices for all tickers
        if isinstance(data.columns, pd.MultiIndex):
            tickers = data.columns.levels[0]
        else:
            # Fallback if structure is unexpected
            return pd.DataFrame()
        
        for ticker in tickers:
            try:
                # Handle case where ticker might not be in top level if download failed for it
                if ticker not in data.columns.levels[0]:
                    continue

                # Get close prices
                # yfinance with auto_adjust=True returns 'Close' as the adjusted close
                if 'Close' in data[ticker]:
                    prices = data[ticker]['Close'].dropna()
                else:
                    continue
                
                if prices.empty:
                    continue
                    
                # Calculate Daily Returns
                daily_returns = prices.pct_change().dropna()
                
                if daily_returns.empty:
                    continue

                # 1. CAGR (Compound Annual Growth Rate)
                days = (prices.index[-1] - prices.index[0]).days
                if days <= 0:
                    continue
                years = days / 365.25
                total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                cagr = (1 + total_return) ** (1 / years) - 1
                
                # 2. Volatility (Annualized Standard Deviation)
                volatility = daily_returns.std() * np.sqrt(252)
                
                # 3. Sharpe Ratio
                # Excess return / Volatility
                sharpe = (cagr - risk_free_rate) / volatility if volatility != 0 else 0
                
                # 4. Max Drawdown
                rolling_max = prices.cummax()
                drawdown = (prices - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                metrics.append({
                    'Ticker': ticker,
                    'CAGR': cagr,
                    'Volatility': volatility,
                    'Sharpe Ratio': sharpe,
                    'Max Drawdown': max_drawdown,
                    'Total Return': total_return
                })
            except Exception as e:
                print(f"Error calculating metrics for {ticker}: {e}")
                
        if not metrics:
            return pd.DataFrame()
            
        return pd.DataFrame(metrics).set_index('Ticker')

    def get_normalized_prices(self, data):
        """
        Returns prices normalized to 100 at the start for comparison plotting.
        """
        normalized = pd.DataFrame()
        
        if isinstance(data.columns, pd.MultiIndex):
            tickers = data.columns.levels[0]
            for ticker in tickers:
                try:
                    if 'Close' in data[ticker]:
                        prices = data[ticker]['Close'].dropna()
                        if not prices.empty:
                            # Normalize to start at 100
                            normalized[ticker] = (prices / prices.iloc[0]) * 100
                except:
                    pass
                
        return normalized
