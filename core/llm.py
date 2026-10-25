"""LLM adapter — optional OpenAI-compatible chat completions.

환경변수로 설정 (모두 선택 사항):
  GEMINI_API_KEY  Google Gemini API 키 (AIza...). 설정 시 Gemini 사용.
  LLM_BASE_URL   OpenAI 호환 엔드포인트 (기본 https://api.openai.com/v1,
                 Gemini 감지 시 https://generativelanguage.googleapis.com/v1beta/openai)
  LLM_API_KEY    API 키 (미설정 시 LLM 비활성 → 규칙 기반 코멘티로 폴백).
                 `AIza` 로 시작하면 Gemini 키로 자동 감지.
  OPENAI_API_KEY LLM_API_KEY 대체 키
  LLM_MODEL      모델명 (기본 gpt-4o-mini, Gemini 감지 시 gemini-1.5-flash)
  LLM_TIMEOUT    요청 타임아웃 초 (기본 45)

Gemini 자동 감지: GEMINI_API_KEY 가 설정되어 있거나 LLM_API_KEY 가 `AIza` 로
시작하면 Google Gemini(공식 OpenAI 호환 엔드포인트)로 자동 전환합니다.
명시적인 LLM_BASE_URL / LLM_MODEL 이 있으면 그 값이 우선합니다.

설계 원칙: 수치/점수/판단은 결정론적으로 계산하고, LLM은 서술(코멘티)만 생성한다.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_DEFAULT_MODEL = "gemini-1.5-flash"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
_GEMINI_KEY_PREFIX = "AIza"


def _clean_key(value: str | None) -> str | None:
    """환경변수로 들어온 API 키에서 따옴표·공백 제거 (붙여넣기 실수 방지)."""
    if value is None:
        return None
    cleaned = value.strip("'\" \t\r\n")
    return cleaned or None


class LLMClient:
    def __init__(self) -> None:
        gemini_key = _clean_key(os.environ.get("GEMINI_API_KEY"))
        llm_key = _clean_key(os.environ.get("LLM_API_KEY"))
        self.api_key = gemini_key or llm_key or _clean_key(os.environ.get("OPENAI_API_KEY"))
        self.is_gemini = bool(gemini_key) or (
            bool(llm_key) and llm_key.startswith(_GEMINI_KEY_PREFIX)
        )
        default_base_url = GEMINI_BASE_URL if self.is_gemini else OPENAI_DEFAULT_BASE_URL
        default_model = GEMINI_DEFAULT_MODEL if self.is_gemini else OPENAI_DEFAULT_MODEL
        self.base_url = os.environ.get("LLM_BASE_URL", default_base_url).rstrip("/")
        self.model = os.environ.get("LLM_MODEL", default_model)
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "45"))
        self.last_error: str | None = None  # 마지막 chat() 실패 사유 (HTTP 상태/본문 등)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, max_tokens: int = 700,
             temperature: float = 0.4, json_mode: bool = False) -> str | None:
        if not self.enabled:
            return None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # 순수 JSON 응답 강제 (OpenAI 호환 엔드포인트 표준 옵션)
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err: str | None = None
        for attempt in range(2):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload, headers=headers, timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except requests.HTTPError as e:
                resp = getattr(e, "response", None)
                code = resp.status_code if resp is not None else "unknown"
                body = ""
                try:
                    if resp is not None:
                        body = (resp.text or "").strip().replace("\n", " ")
                except Exception:
                    body = ""
                if body:
                    body = body[:150]
                last_err = f"HTTP {code}" + (f" · {body}" if body else "")
            except Exception as e:  # 네트워크 오류, 응답 형식 오류 등
                last_err = f"{type(e).__name__}: {e}"[:150]
            if attempt == 0:
                time.sleep(2.0)
        self.last_error = last_err
        return None

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict | None:
        """Chat completion whose content must be a JSON object."""
        self.last_error = None
        text = self.chat(system, user, max_tokens=max_tokens, json_mode=True)
        if not text:
            return None
        parsed = extract_json(text)
        if parsed is None:  # 한 번 더 형식을 강조해 재요청
            text = self.chat(
                system + "\n반드시 JSON 객체만 출력하세요. 마크다운 코드 블록 금지.",
                user, max_tokens=max_tokens, json_mode=True,
            )
            parsed = extract_json(text) if text else None
        return parsed


_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def extract_json(text: str) -> dict | None:
    """LLM 응답 텍스트에서 JSON 객체를 최대한 견고하게 추출한다.

    - 마크다운 코드 블록(```json … ```), 앞뒤 부가 텍스트 제거
    - 후행 쉼표(trailing comma) 정리
    - 성공 시 키를 소문자로 정규화해 반환
    """
    if not text:
        return None
    t = text.strip()
    # 코드 블록 제거
    t = re.sub(r"```(?:json)?\s*|```", "", t).strip()
    for candidate in (t, _first_brace_block(t)):
        if not candidate:
            continue
        # 후행 쉼표 정리 ({"a": 1,} → {"a": 1})
        cleaned = _TRAILING_COMMA.sub(r"\1", candidate)
        for cand in (cleaned, candidate):
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return {_lower(k): v for k, v in obj.items()}
            except Exception:
                continue
    return None


def _lower(key: str) -> str:
    return key.lower().strip() if isinstance(key, str) else key


def _first_brace_block(t: str) -> str | None:
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
    return None


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
