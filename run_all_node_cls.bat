@echo off
chcp 65001 >nul
title DHG-Bench FULL RE-RUN by DATASET (live output)
cd /d "%~dp0"
"D:\anaconda3\envs\dhgbench\python.exe" -u "%~dp0run_all_node_cls.py"
echo.
echo ============================================
echo Full re-run finished. Press any key to close.
pause >nul
