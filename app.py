import sys
import os

# Add the subdirectory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Fund Recommendation System'))

# Import and run the main app
import streamlit as st

# Set page config
st.set_page_config(page_title="Fund RecSys AI", layout="wide")

# Import the actual app logic
exec(open('Fund Recommendation System/recsys_app.py', encoding='utf-8').read())
