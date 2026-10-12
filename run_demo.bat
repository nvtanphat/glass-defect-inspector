@echo off
title Glass Defect Inspector
echo ===================================================
echo Starting Glass Defect Inspector. Apply database migrations first.
echo ===================================================
cd /d "%~dp0"
if not defined INSPECTOR_PORT set "INSPECTOR_PORT=8501"
echo Open http://127.0.0.1:%INSPECTOR_PORT% after the server is ready.
python -m glass_inspector
pause
