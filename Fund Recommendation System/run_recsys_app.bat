@echo off
echo ========================================
echo 啟動推薦系統 Streamlit App
echo ========================================
echo.

cd /d "%~dp0"

echo 正在啟動應用...
echo 應用會在瀏覽器自動開啟
echo 按 Ctrl+C 可以停止應用
echo.

.\.venv\Scripts\python.exe -m streamlit run recsys_app.py

pause
