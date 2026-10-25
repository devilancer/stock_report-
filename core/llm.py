"""LLM adapter — optional OpenAI-compatible chat completions.

환경변수로 설정 (모두 선택 사항):
  LLM_BASE_URL   OpenAI 호환 엔드포인트 (기본 https://api.openai.com/v1)
  LLM_API_KEY    API 키 (미설정 시 LLM 비활성 → 규칙 기반 코멘티로 폴백)
  OPENAI_API_KEY LLM_API_KEY 대체 키
  LLM_MODEL      모델명 (기본 gpt-4o-mini)
  LLM_TIMEOUT    요청 타임아웃 초 (기본 45)

설계 원칙: 수치/점수/판단은 결정론적으로 계산하고, LLM은 서술(코멘티)만 생성한다.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests


class LLMClient:
    def __init__(self) -> None:
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "45"))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, max_tokens: int = 700,
             temperature: float = 0.4) -> str | None:
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
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(2):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload, headers=headers, timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception:
                if attempt == 0:
                    time.sleep(2.0)
        return None

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict | None:
        """Chat completion whose content must be a JSON object."""
        text = self.chat(system, user, max_tokens=max_tokens)
        if not text:
            return None
        parsed = extract_json(text)
        if parsed is None:  # 한 번 더 형식을 강조해 재요청
            text = self.chat(
                system + "\n반드시 JSON 객체만 출력하세요. 마크다운 코드 블록 금지.",
                user, max_tokens=max_tokens,
            )
            parsed = extract_json(text) if text else None
        return parsed


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    # 코드 블록 제거
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.M).strip()
    for candidate in (t, _first_brace_block(t)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _first_brace_block(t: str) -> str | None:
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(t[start:], start):
        if ch == "{":
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
