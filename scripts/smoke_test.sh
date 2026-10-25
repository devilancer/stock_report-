#!/usr/bin/env bash
# 스모크 테스트: 헬스체크 + 실제 분석 스트리밍 1건 검증
# usage: bash scripts/smoke_test.sh https://desk.example.com [SYMBOL]
set -euo pipefail

BASE="${1:?base url required}"
BASE="${BASE%/}"
SYMBOL="${2:-AAPL}"

echo "[smoke] health: $BASE/api/health"
for i in $(seq 1 15); do
  if curl -sf --max-time 5 "$BASE/api/health" | grep -q '"ok"'; then
    echo "[smoke] health OK"
    break
  fi
  if [ "$i" = 15 ]; then
    echo "[smoke] health FAILED after 15 attempts"
    exit 1
  fi
  sleep 4
done

echo "[smoke] analysis stream: $SYMBOL"
OUT="$(curl -sN --max-time 240 "$BASE/api/analyze/stream?symbol=$SYMBOL")"
if ! echo "$OUT" | grep -q '"type": "final"'; then
  echo "[smoke] analysis FAILED — 마지막 이벤트:"
  echo "$OUT" | tail -6
  exit 1
fi
VERDICT="$(echo "$OUT" | grep -o '"verdict": "[^"]*"' | tail -1)"
SCORE="$(echo "$OUT" | grep -o '"score": [0-9.]*' | tail -1)"
echo "[smoke] analysis OK — ${VERDICT}, ${SCORE}"
