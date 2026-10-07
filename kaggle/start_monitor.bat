@echo off
title Kaggle Auto-Resume Monitor (2h interval)
cd /d "%~dp0\.."
echo Starting Kaggle Auto-Resume Monitor...
python kaggle/auto_resume_monitor.py --interval 7200
pause
