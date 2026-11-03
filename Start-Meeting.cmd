@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\Start-Meeting.ps1"
if errorlevel 1 (
    echo.
    echo Meeting startup did not complete. See the error above.
    pause
)
