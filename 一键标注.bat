@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================================
echo  合照标注 + 人数统计
echo ============================================================
echo.
".venv\Scripts\python.exe" mark_photo.py %*
echo.
pause
