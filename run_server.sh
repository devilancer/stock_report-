#!/usr/bin/env bash
# Stock Desk server control: bash run_server.sh [start|stop|restart]
set -e
cd "$(dirname "$0")"
PIDFILE=".server.pid"

stop() {
  if [ -f "$PIDFILE" ]; then
    OLD=$(cat "$PIDFILE")
    if [ -n "$OLD" ] && [ -d "/proc/$OLD" ]; then
      kill "$OLD" 2>/dev/null || true
      sleep 1
    fi
    rm -f "$PIDFILE"
  fi
}

start() {
  nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 8787 > .openclaw-run.log 2>&1 &
  echo $! > "$PIDFILE"
  sleep 2
  curl -s http://127.0.0.1:8787/api/health && echo " (pid $(cat "$PIDFILE"))"
}

case "${1:-restart}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
esac
