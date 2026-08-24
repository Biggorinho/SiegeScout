@echo off
cd /d "%~dp0"

:: Re-launch as admin if not already elevated (required for Npcap packet capture)
net session >nul 2>&1
if %errorLevel% == 0 goto :run
echo Requesting administrator privileges for live stats (Npcap)...
powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs -WorkingDirectory '%~dp0'"
exit /b

:run
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python gui.py
pause
