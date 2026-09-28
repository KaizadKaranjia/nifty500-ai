@echo off
REM Nifty 500 AI daily job - wrapper for Windows Task Scheduler.
REM Uses the virtual environment .venv next to this file; output is appended to logs\cron.log
cd /d "%~dp0"
if not exist logs mkdir logs
echo ===== %date% %time% start ===== >> logs\cron.log
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe nifty500_daily.py %* >> logs\cron.log 2>&1
) else (
    python nifty500_daily.py %* >> logs\cron.log 2>&1
)
echo ===== %date% %time% end (exit %errorlevel%) ===== >> logs\cron.log
