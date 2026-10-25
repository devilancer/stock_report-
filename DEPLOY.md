# Stock Desk 배포 가이드

세 가지 방법이 있습니다. **A. 지금 바로 공개 URL(임시 터널)** · **B. Docker로 자체 서버/VPS** · **C. PaaS(Render/Railway/Fly)**

---

## A. 임시 공개 URL (Cloudflare Quick Tunnel)

별도 계정 없이 즉시 공개 URL을 받는 방법입니다. 터널 프로세스가 살아있는 동안만 유효합니다.

```bash
curl -L -o /usr/local/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared

cd stock-desk
bash run_server.sh restart                      # 8787 포트 기동
cloudflared tunnel --url http://127.0.0.1:8787  # trycloudflare.com URL 발급
```

`https://xxx-yyy.trycloudflare.com` 형태의 URL이 출력되면 그대로 배포 완료입니다.
중요: LLM 키가 필요하면 터널을 띄우기 전에 `export LLM_API_KEY=...` 후 서버를 재시작하세요.

## B. Docker (자체 서버 / VPS)

```bash
cd stock-desk
cp .env.example .env         # LLM 사용 시 편집
docker compose up -d --build
curl http://<서버IP>:8787/api/health
```

- 이미지 빌드 → `stock-desk:latest`, `restart: unless-stopped`로 재기동 보장
- 상태 확인: `docker compose logs -f`, 헬스체크는 Dockerfile에 내장
- 포트는 8787. 방화벽에서 해당 포트를 열거나 nginx/caddy로 리버스 프록시하세요.

### nginx 리버스 프록시 예시 (SSE 스트리밍 필수 옵션 포함)

```nginx
server {
  listen 443 ssl;
  server_name desk.example.com;

  location / {
    proxy_pass http://127.0.0.1:8787;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;          # SSE 필수
    proxy_read_timeout 300s;      # 분석 완료까지 대기
    proxy_set_header X-Forwarded-For $proxy_x_forwarded_for;
  }
}
```

## C. PaaS

**Render** — `render.yaml` 포함되어 있음: 저장소 연결 후 Blueprint로 배포하면 무료 티어(`plan: free`)가 자동 적용됩니다.
"New +" → "Blueprint" 선택 후 저장소를 연결하거나, "New Web Service"로 진행해도 Render는 Dockerfile을 자동 감지.

- 무료 플랜: 인스턴스 스펙은 무료 티어 기본값(공유 CPU / 512MB RAM)으로 기동되며, 비활성 시 15분 후 슬립했다가 요청 시 다시 깨어납니다(첫 응답이 약간 느려질 수 있음).
- `GEMINI_API_KEY` 등록(권장): Render 대시보드에서 서비스 선택 → **Environment** 탭 → `Add Environment Variable` → Key에 `GEMINI_API_KEY`, Value에 [Google AI Studio](https://aistudio.google.com/apikey)에서 발급한 API 키 입력 → **Save Changes** (재배포 자동 실행).
  - `GEMINI_API_KEY`를 넣으면 `LLM_MODEL`(기본값 `gemini-2.5-flash`)과 `LLM_BASE_URL`(Gemini OpenAI 호환성 엔드포인트)은 `render.yaml` 기본값을 그대로 사용하면 됩니다.
- `LLM_API_KEY` (선택): Gemini 대신 다른 OpenAI 호환 LLM을 쓸 때만 같은 방법으로 Environment에 등록하세요. 이 경우 `LLM_MODEL`과 `LLM_BASE_URL`도 해당 서비스에 맞게 변경해야 합니다.
- 두 키 모두 대시보드에서만 입력 가능합니다(`sync: false`) — Blueprint 배포 시 값을 미리 채우지 않습니다.

**Railway** — `railway up` 또는 GitHub 연결. Start Command: `gunicorn app:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT -w 2 --timeout 180`.

**Fly.io** — `fly launch --dockerfile Dockerfile`, `fly secrets set LLM_API_KEY=...`, `fly deploy`.

---

## D. 배포 자동화 (GitHub Actions)

이 디렉터리를 GitHub 저장소 루트로 푸시하면 3개 워크플로가 동작합니다.

| 워크플로 | 트리거 | 하는 일 |
|---------|--------|---------|
| `ci.yml` | push / PR | 오프라인 테스트 11개 + 문법 검사 + Docker 빌드 검증 |
| `deploy-docker.yml` | main 푸시 / `v*` 태그 | GHCR 이미지 빌드·푸시 (`ghcr.io/<owner>/<repo>:latest`, `:sha-*`, `:X.Y.Z`) + 컨테이너 기동 헬스 게이트 |
| `deploy-ssh.yml` | main 푸시 / 수동 실행 | 서버에서 git pull → `docker compose up -d --build` → 공개 URL 스모크 테스트 |

### 시크릿 설정 (Settings → Secrets and variables → Actions)

| 시크릿 | 필수 | 설명 |
|--------|------|------|
| `SSH_HOST` / `SSH_USER` / `SSH_KEY` | deploy-ssh용 | 서버 접속 정보 (KEY는 PEM 개인키) |
| `SMOKE_URL` | deploy-ssh용 | 공개 URL (예: https://desk.example.com) |
| `SSH_PORT` / `DEPLOY_DIR` / `SMOKE_SYMBOL` | 선택 | 기본값 22 / `/srv/stock-desk` / AAPL |
| `LLM_API_KEY` | 선택 | 서버 측 `.env`로 관리하는 것을 권장 |

### 릴리스 절차

```bash
bash scripts/tag_release.sh 1.2.0   # v1.2.0 태그 → 이미지 빌드 자동 실행
```

### 로컬에서 CI와 동일하게 확인

```bash
python -m unittest discover -s tests -v   # 오프라인 테스트 (네트워크 불필요)
bash scripts/smoke_test.sh http://127.0.0.1:8787 AAPL
```

---

## 운영 체크리스트

- [ ] `GET /api/health` → `{"ok": true}` 응답 확인
- [ ] 실제 종목 1건 분석 스트리밍 정상 (AAPL / 005930.KS / 600519)
- [ ] LLM 사용 시 `.env`의 키가 코드/이미지에 포함되지 않도록 env_file/시크릿 매니저로 주입
- [ ] 공개 노출 시 반드시 HTTPS + 최소한의 접근 통제(basic auth, IP allowlist, VPN)
- [ ] 데이터 소스는 공개 API(Yahoo/akshare) — 과도한 호출은 소스 측 레이트리밋에 걸릴 수 있음

## 아키텍처 메모

- `app.py` FastAPI + SSE(`/api/analyze/stream`) — 분석은 요청별 스레드에서 실행, 이벤트 스트림으로 진행 상황 전달
- `core/agents.py` 4개 에이전트 체인: 점수/판단/포지션은 결정론적, LLM은 서술만 생성(미설정 시 규칙 기반 폴백)
- 정적 파일은 FastAPI가 직접 서빙 — 프론트엔드 빌드 도구 불필요
