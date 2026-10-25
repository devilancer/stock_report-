#!/usr/bin/env bash
# Stock Desk keep-alive — 서버(8787)와 공개 터널이 죽으면 재기동합니다.
# 워크스페이스 전용 스크립트. Quick Tunnel 재기동 시 공개 URL이 바뀔 수 있습니다
# (변경된 URL은 .openclaw/tmp/public_url.txt 에 기록됨).
set -u
cd "$(dirname "$0")"
W="$(cd .. && pwd)"
CF="$W/.openclaw/tmp/cloudflared"
PIDF="$W/.openclaw/tmp/tunnel.pid"
URLF="$W/.openclaw/tmp/public_url.txt"

while true; do
  if ! curl -s --max-time 5 http://127.0.0.1:8787/api/health 2>/dev/null | grep -q '"ok"'; then
    echo "[supervisor] $(date '+%F %T') server down → restart" >> .openclaw-run.log
    bash run_server.sh restart >> .openclaw-run.log 2>&1
  fi

  TPID="$(cat "$PIDF" 2>/dev/null || true)"
  if [ -z "$TPID" ] || [ ! -d "/proc/$TPID" ]; then
    echo "[supervisor] $(date '+%F %T') tunnel down → restart" >> .openclaw-run.log
    nohup "$CF" tunnel --url http://127.0.0.1:8787 --no-autoupdate \
      > "$W/.openclaw/tmp/tunnel.log" 2>&1 &
    echo $! > "$PIDF"
    sleep 8
    grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$W/.openclaw/tmp/tunnel.log" \
      | head -1 > "$URLF"
  fi

  sleep 30
done
