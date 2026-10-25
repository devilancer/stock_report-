# STOCK DESK — 4-AGENT 리서치 파이프라인

> 저장소: `devilancer/stock_report-` · 프로토타입(`ai-주식-분석-워크플로.tsx`)은 루트에 보존되어 있습니다.

종목 코드 하나로 **펀더멘털 → 테크니컬 → 시장 리서치 → 포트폴리오 매니저** 4개의 AI 에이전트 체인을
순차 실행해, 판단 근거가 남는 큐레이티드 주식 분석 보고서를 생성하는 웹 앱입니다.

## 파이프라인

| # | 에이전트 | 담당 | 산출 |
|---|---------|------|------|
| 00 | 데이터 수집 | 가격(2년 일봉) · 재무 지표 · 시장 지수/컨텍스트 | 정규화 데이터셋 |
| 01 | 펀더멘털 애널리스트 | PER/PBR/ROE/마진/성장/부채/배당 | 점수 + 노트 |
| 02 | 테크니컬 애널리스트 | 추세·RSI·MACD·변동성·52주 레인지·거래량 | 점수 + 노트 |
| 03 | 시장 리서치 애널리스트 | 시장 레짐·상대강도·VIX/변동성 환경 | 점수 + 노트 |
| 04 | 포트폴리오 매니저 | 3개 노트 가중 종합 → 매수/매도 결정 | 최종 보고서 |

- 시장 레짐(25%) + 펀더멘털(36%) + 테크니컬(29%) 가중 종합, 데이터 커버리지 보정
- 포지션 크기 = 기본 6~8% × 변동성 타기팅(목표 연율 16%) × 판단 신뢰도, 상한 12%
- 손절/목표가는 ATR 기반 리스크:리워드 1:1.5 / 1:2.6 구조
- 보고서는 `.md` 다운로드 가능

## LLM 연동 (선택)

점수·판단·포지션 계산은 **결정론적 규칙**으로 고정하고, LLM은 각 에이전트의
서술(요약·근거·경고)만 생성합니다. 키 미설정 또는 호출 실패 시 규칙 기반 코멘티로 자동 폴백합니다.

```bash
export LLM_API_KEY=sk-...                 # 또는 OPENAI_API_KEY
export LLM_MODEL=gpt-4o-mini              # 기본값
export LLM_BASE_URL=https://api.openai.com/v1   # OpenAI 호환 엔드포인트 (로컬/다른 벤더 가능)
export LLM_TIMEOUT=45
bash run_server.sh restart
```

- 보고서/노트에 코멘티 소스 배지(`LLM 코멘티` / `규칙 기반 코멘티`)가 표시됩니다.
- OpenAI 호환 API라면 누구나 연동 가능: OpenAI, Azure, vLLM, Ollama(`/v1`), LM Studio 등.

### Google Gemini 사용

`GEMINI_API_KEY`만 설정하면 끝입니다 — Gemini 엔드포인트와 기본 모델이 자동 구성됩니다.

```bash
export GEMINI_API_KEY=AIza...             # https://aistudio.google.com/apikey
bash run_server.sh restart
```

- `GEMINI_API_KEY`가 있거나 `LLM_API_KEY`가 `AIza`로 시작하면 자동으로 Gemini(공식 OpenAI 호환
  엔드포인트)를 사용합니다.
- 자동 구성값: `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`,
  `LLM_MODEL=gemini-1.5-flash`
- `LLM_BASE_URL` / `LLM_MODEL`을 직접 설정하면 그 값이 우선합니다.

## 지원 시장

- 미국: `AAPL`, `NVDA` (Yahoo Finance)
- 한국: `005930.KS`, `000660.KQ` (Yahoo Finance, `.KS`/`.KQ` 접미사)
- 중국 A주: `600519`, `000001` (6자리 코드, Yahoo + akshare 보강)

## 실행

```bash
pip install -r requirements.txt
python3 app.py            # http://localhost:8787
```

## 구조

```
app.py                 FastAPI + SSE(/api/analyze/stream)
core/data.py           시장 감지 · 가격/재무/지수 수집 정규화
core/indicators.py     RSI/MACD/볼린저/ATR 등 지표
core/llm.py            OpenAI 호환 LLM 어댑터(JSON 코멘티 + 폴백)
core/agents.py         4개 에이전트 + 파이프라인
static/                에디토리얼 스타일 프론트엔드(워크플로우 UI + 보고서)
```

> 면책: 공개 데이터 기반 자동 생성 리서치이며 투자 권유가 아닙니다.
