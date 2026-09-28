#!/bin/bash
# Cron wrapper for the Nifty 500 AI daily job.
# Runs with low CPU priority so Oracle and other services on the server are not disturbed.
# Extra arguments are passed on, e.g.:  ./run_daily.sh --status
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR" || exit 1
mkdir -p "$APP_DIR/logs"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
PY="$APP_DIR/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3.12 || command -v python3)"
nice -n 10 "$PY" "$APP_DIR/nifty500_daily.py" "$@" >> "$APP_DIR/logs/cron.log" 2>&1
rc=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') exit code $rc" >> "$APP_DIR/logs/cron.log"
exit $rc
