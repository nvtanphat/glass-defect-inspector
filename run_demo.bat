@echo off
title Glass Defect Inspector Demo
echo ===================================================
echo Starting Glass Defect Inspector Demo...
echo ===================================================
cd /d "%~dp0"
start http://localhost:8501
python app.py
pause
