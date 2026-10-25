#!/usr/bin/env bash
# 릴리스 태그 생성 → GHCR 이미지 빌드 워크플로가 자동 실행됩니다.
# usage: bash scripts/tag_release.sh 1.2.0
set -euo pipefail

V="${1:?version required (e.g. 1.2.0)}"
V="${V#v}"

git tag -a "v$V" -m "release v$V"
git push origin "v$V"
echo "[release] v$V pushed — Docker Image 워크플로가 실행됩니다."
