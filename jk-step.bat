@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install_windows.bat first.
  exit /b 1
)
".venv\Scripts\python.exe" "jk_step.py" %*
