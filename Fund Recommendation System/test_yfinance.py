import yfinance as yf
import pandas as pd
import certifi
import os
import shutil

# 1. Get the current cert path
current_cert = certifi.where()
print(f"Original Cert Path: {current_cert}")

# 2. Define a safe path (ASCII only)
# Using the user's home directory or a temp folder
safe_cert_path = os.path.join(os.path.expanduser("~"), ".gemini", "cacert.pem")
os.makedirs(os.path.dirname(safe_cert_path), exist_ok=True)

# 3. Copy the cert
try:
    shutil.copy(current_cert, safe_cert_path)
    print(f"Copied cert to: {safe_cert_path}")
except Exception as e:
    print(f"Failed to copy cert: {e}")

# 4. Set the environment variable
os.environ['CURL_CA_BUNDLE'] = safe_cert_path
print(f"Set CURL_CA_BUNDLE to: {os.environ['CURL_CA_BUNDLE']}")

print("Testing yfinance...")
tickers = ["SPY"]
print(f"Attempting to download: {tickers}")

try:
    # We must NOT pass session if we want yfinance to use its internal curl
    data = yf.download(tickers, period="1mo", auto_adjust=True)
    print("\nDownload finished.")
    
    if data.empty:
        print("Result: EMPTY DataFrame.")
    else:
        print("Result: SUCCESS.")
        print(data.head())
        
except Exception as e:
    print(f"\nResult: FAILED with error: {e}")

print("\nTesting complete.")
