@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*
if errorlevel 1 (
  echo Installation failed. Review the error above; running again resumes downloads.
  pause
  exit /b 1
)
pause
