#!/usr/bin/env bash
# 서버에서 실행하는 배포 스크립트 (git pull → 도커 재기동 → 스모크)
# usage: SMOKE_URL=https://desk.example.com bash scripts/server_deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="${BRANCH:-main}"
SMOKE_URL="${SMOKE_URL:-http://127.0.0.1:8787}"
SMOKE_SYMBOL="${SMOKE_SYMBOL:-AAPL}"

echo "[deploy] git pull ($BRANCH)"
git pull --ff-only origin "$BRANCH"

echo "[deploy] docker compose up -d --build"
docker compose up -d --build

echo "[deploy] smoke test"
bash scripts/smoke_test.sh "$SMOKE_URL" "$SMOKE_SYMBOL"

echo "[deploy] done"
