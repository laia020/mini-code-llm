@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "%~dp0iniciar.py" %*
    goto finished
)

py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    py -3.11 "%~dp0iniciar.py" %*
    goto finished
)

python --version >nul 2>&1
if not errorlevel 1 (
    python "%~dp0iniciar.py" %*
    goto finished
)

echo Instale Python 3.11 antes de iniciar o projeto:
echo https://www.python.org/downloads/
pause
exit /b 1

:finished
set "launcher_exit=%errorlevel%"
if not "%launcher_exit%"=="0" pause
exit /b %launcher_exit%
