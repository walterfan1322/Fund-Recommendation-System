import yfinance as yf
import pandas as pd
import certifi
import os
import shutil

# SSL Fix
try:
    current_cert = certifi.where()
    safe_cert_path = os.path.join(os.path.expanduser("~"), ".gemini", "cacert.pem")
    os.makedirs(os.path.dirname(safe_cert_path), exist_ok=True)
    if not os.path.exists(safe_cert_path):
        shutil.copy(current_cert, safe_cert_path)
    os.environ['CURL_CA_BUNDLE'] = safe_cert_path
except:
    pass

print("Testing Mutual Funds...")
tickers = ["PGNAX", "FSELX"] # Allianz AI (US), Fidelity Semi
print(f"Attempting to download: {tickers}")

try:
    data = yf.download(tickers, period="1mo", auto_adjust=True)
    print("\nDownload finished.")
    
    if data.empty:
        print("Result: EMPTY DataFrame.")
    else:
        print("Result: SUCCESS.")
        print(data.head())
        
except Exception as e:
    print(f"\nResult: FAILED with error: {e}")
