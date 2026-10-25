"""Stock Desk 오프라인 테스트 — 네트워크/LLM 없이 항상 통과해야 하는 핵심 로직.

실행: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import indicators as ta                      # noqa: E402
from core.agents import (                             # noqa: E402
    FundamentalAgent,
    MarketResearchAgent,
    PortfolioManagerAgent,
    TechnicalAgent,
    _llm_note,
)
from core.data import DataCollector, StockData, detect_market, normalize_symbol  # noqa: E402
from core.llm import extract_json                     # noqa: E402


def make_data(seed: int = 42) -> StockData:
    """결정론적 합성 데이터로 StockData 생성 (고정 seed)."""
    rng = np.random.default_rng(seed)
    n = 300
    rets = rng.normal(0.0006, 0.015, n)
    close = 100 * np.exp(np.cumsum(rets))  # ndarray — 인덱스 정렬 이슈 방지
    df = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.006, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=pd.bdate_range("2025-01-01", periods=n))
    df = DataCollector()._add_ta(df)
    assert np.isfinite(df["close"].iloc[-1])

    profile = {
        "name": "테스트 종목", "sector": "Technology", "industry": "Semiconductors",
        "market_cap": 5e11, "pe_ttm": 18.5, "pe_forward": 15.2, "pb": 2.4,
        "roe": 0.16, "gross_margin": 0.45, "net_margin": 0.12,
        "revenue_growth": 0.14, "earnings_growth": 0.18,
        "debt_to_equity": 55.0, "dividend_yield": 1.2, "beta": 1.1,
        "week52_high": float(df["high"].tail(252).max()),
        "week52_low": float(df["low"].tail(252).min()),
    }
    index = df[["open", "high", "low", "close", "volume"]].copy()
    return StockData(
        symbol="TEST", yahoo_symbol="TEST", market="us", name="테스트 종목",
        currency="USD", prices=df, profile=profile,
        index_prices=index, index_name="S&P 500", vol_index=None, data_issues=[],
    )


class TestSymbolRouting(unittest.TestCase):
    def test_detect_market(self):
        self.assertEqual(detect_market("600519"), "cn")
        self.assertEqual(detect_market("005930.KS"), "kr")
        self.assertEqual(detect_market("000660.KQ"), "kr")
        self.assertEqual(detect_market("AAPL"), "us")

    def test_normalize_symbol(self):
        self.assertEqual(normalize_symbol("600519"), "600519.SS")   # 상해
        self.assertEqual(normalize_symbol("000001"), "000001.SZ")   # 심천
        self.assertEqual(normalize_symbol("900901"), "900901.SS")
        self.assertEqual(normalize_symbol("aapl"), "AAPL")


class TestIndicators(unittest.TestCase):
    def test_sma_matches_manual_mean(self):
        s = pd.Series(np.arange(1, 31, dtype=float))
        self.assertAlmostEqual(float(ta.sma(s, 5).iloc[-1]), float(s.tail(5).mean()))

    def test_rsi_bounds(self):
        s = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 1, 120)) + 100)
        r = ta.rsi(s, 14).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())

    def test_macd_identity(self):
        s = pd.Series(np.random.default_rng(2).normal(0, 1, 200).cumsum() + 50)
        line, sig, hist = ta.macd(s)
        self.assertTrue(np.allclose(hist.iloc[-10:], (line - sig).iloc[-10:]))

    def test_drawdown_non_positive(self):
        s = pd.Series([100, 110, 90, 95, 120])
        self.assertLessEqual(ta.max_drawdown(s), 0)


class TestLLMAdapter(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(extract_json('```json\n{"summary": "ok", "bullets": [], "risks": []}\n```')["summary"], "ok")
        self.assertEqual(extract_json('설명 {"summary": "brace", "bullets": [], "risks": []} 끝')["summary"], "brace")
        self.assertIsNone(extract_json("완전히 다른 텍스트"))
        self.assertIsNone(extract_json('{"summary": "broken", "bullets": ['))

    def test_extract_json_trailing_comma(self):
        parsed = extract_json('{"summary": "tc", "bullets": ["a",], "risks": [],}')
        self.assertEqual(parsed["summary"], "tc")
        self.assertEqual(parsed["bullets"], ["a"])

    def test_extract_json_surrounding_prose(self):
        text = '다음은 분석 결과입니다:\n```json\n{"Summary": "x", "Bullets": [], "Risks": []}\n```\n추가 설명 텍스트입니다.'
        parsed = extract_json(text)
        self.assertEqual(parsed["summary"], "x")

    def test_extract_json_key_case_normalized_to_lower(self):
        parsed = extract_json('{"Summary": "ok", "Bullets": ["b"], "Risks": ["r"]}')
        self.assertIn("summary", parsed)
        self.assertEqual(parsed["summary"], "ok")
        self.assertEqual(parsed["bullets"], ["b"])
        self.assertEqual(parsed["risks"], ["r"])

    def test_extract_json_braces_inside_strings(self):
        text = '참고 {"summary": "JSON은 { 중괄호 } 포함", "bullets": [], "risks": []} 끝'
        parsed = extract_json(text)
        self.assertEqual(parsed["summary"], "JSON은 { 중괄호 } 포함")

    def test_api_key_cleaned_of_quotes_and_whitespace(self):
        import os
        saved = {}
        keys = ("GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
        for k in keys:
            saved[k] = os.environ.pop(k, None)
        try:
            os.environ["LLM_API_KEY"] = '  "sk-test-key-123" \n'
            from core.llm import LLMClient
            c = LLMClient()
            self.assertEqual(c.api_key, "sk-test-key-123")
            self.assertFalse(c.is_gemini)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_gemini_key_with_quotes_still_detected(self):
        import os
        saved = {}
        keys = ("GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
        for k in keys:
            saved[k] = os.environ.pop(k, None)
        try:
            os.environ["GEMINI_API_KEY"] = "' AIzaSyQuoted '"
            from core.llm import LLMClient
            c = LLMClient()
            self.assertTrue(c.is_gemini)
            self.assertEqual(c.api_key, "AIzaSyQuoted")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_chat_records_http_error_in_last_error(self):
        import os
        from unittest import mock
        import requests as requests_mod
        from core.llm import LLMClient

        saved = {}
        keys = ("GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
        for k in keys:
            saved[k] = os.environ.pop(k, None)
        try:
            os.environ["LLM_API_KEY"] = "sk-test"
            c = LLMClient()
            resp = requests_mod.Response()
            resp.status_code = 401
            resp._content = b'{"error": {"message": "Invalid API key"}}'
            err = requests_mod.HTTPError("401 Client Error", response=resp)
            with mock.patch("core.llm.requests.post", side_effect=err), \
                 mock.patch("core.llm.time.sleep"):
                out = c.chat("sys", "user")
            self.assertIsNone(out)
            self.assertIsNotNone(c.last_error)
            self.assertIn("401", c.last_error)
            self.assertIn("Invalid API key", c.last_error)
            # complete_json도 실패 사유를 보존하는지 확인
            c.last_error = None
            with mock.patch("core.llm.requests.post", side_effect=err), \
                 mock.patch("core.llm.time.sleep"):
                parsed = c.complete_json("sys", "user")
            self.assertIsNone(parsed)
            self.assertIn("401", c.last_error)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_chat_sends_json_mode_response_format(self):
        import os
        from unittest import mock
        from core.llm import LLMClient

        saved = {}
        keys = ("GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
        for k in keys:
            saved[k] = os.environ.pop(k, None)
        try:
            os.environ["LLM_API_KEY"] = "sk-test"
            c = LLMClient()
            captured = {}

            class FakeResp:
                status_code = 200
                def raise_for_status(self):
                    pass
                def json(self):
                    return {"choices": [{"message": {"content": '{"summary": "ok"}'}}]}

            def fake_post(url, json=None, headers=None, timeout=None):
                captured["payload"] = json
                return FakeResp()

            with mock.patch("core.llm.requests.post", side_effect=fake_post):
                plain = c.chat("sys", "user")
            self.assertNotIn("response_format", captured["payload"])
            with mock.patch("core.llm.requests.post", side_effect=fake_post):
                parsed = c.complete_json("sys", "user")
            self.assertEqual(captured["payload"].get("response_format"), {"type": "json_object"})
            self.assertEqual(parsed["summary"], "ok")
            self.assertIsNone(c.last_error)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_llm_disabled_falls_back_to_rule(self):
        import os
        os.environ.pop("LLM_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        nar, source = _llm_note("t", "역할", 50.0, "중립", "facts", lambda e: None)
        self.assertIsNone(nar)
        self.assertEqual(source, "rule")


class TestLLMGeminiDetection(unittest.TestCase):
    """GEMINI_API_KEY / AIza 프리픽스 감지 → 엔드포인트·기본 모델 자동 구성."""

    def _client_with_env(self, **env):
        import os
        saved = {}
        keys = ("GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
        for k in keys:
            saved[k] = os.environ.pop(k, None)
        try:
            for k, v in env.items():
                if v is not None:
                    os.environ[k] = v
            from core.llm import LLMClient
            return LLMClient()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_gemini_key_sets_gemini_endpoint_and_model(self):
        c = self._client_with_env(GEMINI_API_KEY="AIzaSyTest123")
        self.assertTrue(c.is_gemini)
        self.assertEqual(c.api_key, "AIzaSyTest123")
        self.assertEqual(c.base_url, "https://generativelanguage.googleapis.com/v1beta/openai")
        self.assertEqual(c.model, "gemini-3.5-flash-lite")

    def test_llm_key_with_aiza_prefix_detected_as_gemini(self):
        c = self._client_with_env(LLM_API_KEY="AIzaSyABC-def_123")
        self.assertTrue(c.is_gemini)
        self.assertEqual(c.api_key, "AIzaSyABC-def_123")
        self.assertEqual(c.base_url, "https://generativelanguage.googleapis.com/v1beta/openai")
        self.assertEqual(c.model, "gemini-3.5-flash-lite")

    def test_legacy_gemini_models_are_remapped_to_latest(self):
        """구버전 모델명(1.5/2.5/2.0) 명시 시 최신 모델로 자동 치환 (안전장치)."""
        for legacy in ("gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"):
            c = self._client_with_env(GEMINI_API_KEY="AIzaSyTest123", LLM_MODEL=legacy)
            self.assertTrue(c.is_gemini)
            self.assertEqual(c.model, "gemini-3.5-flash-lite")

    def test_gemini_404_falls_back_to_supported_model(self):
        """호출 중 404(모델 미지원) → 경량 폴백 모델로 한 번 재시도 후 성공."""
        import json as _json
        import os
        from unittest import mock

        import requests as _requests

        saved = {k: os.environ.pop(k, None) for k in
                 ("GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL")}
        try:
            os.environ["GEMINI_API_KEY"] = "AIzaSyTest123"
            os.environ["LLM_MODEL"] = "gemini-3.8-flash"  # 폴백 경로 검증용: 무거운 모델 명시
            from core.llm import LLMClient
            c = LLMClient()
            self.assertEqual(c.model, "gemini-3.8-flash")

            def fake_post(url, **kwargs):
                sent_model = kwargs["json"]["model"]
                if sent_model == "gemini-3.8-flash":
                    resp = mock.Mock()
                    resp.status_code = 404
                    resp.text = _json.dumps({
                        "error": {"code": 404, "message":
                                  "models/gemini-3.8-flash is not found for API version v1beta"}
                    })
                    raise _requests.HTTPError(response=resp)
                ok = mock.Mock()
                ok.status_code = 200
                ok.json.return_value = {
                    "choices": [{"message": {"content": f"ok:{sent_model}"}}]
                }
                return ok

            with mock.patch.object(_requests, "post", side_effect=fake_post), \
                 mock.patch("core.llm.time.sleep") as m_sleep:
                out = c.chat("sys", "user")
            self.assertEqual(out, "ok:gemini-3.5-flash-lite")  # 폴백 모델로 재시도 성공
            self.assertEqual(c.model, "gemini-3.5-flash-lite")  # 성공 모델로 유지
            self.assertIsNone(c.last_error)
            m_sleep.assert_not_called()  # 폴백은 즉시 재시도 (sleep 없음)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_gemini_404_both_models_fails_gracefully(self):
        """폴백 모델까지 404면 None 을 반환하고 마지막 오류를 보존한다."""
        import json as _json
        import os
        from unittest import mock

        import requests as _requests

        saved = {k: os.environ.pop(k, None) for k in
                 ("GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL")}
        try:
            os.environ["GEMINI_API_KEY"] = "AIzaSyTest123"
            from core.llm import LLMClient
            c = LLMClient()

            def fake_post(url, **kwargs):
                sent_model = kwargs["json"]["model"]
                resp = mock.Mock()
                resp.status_code = 404
                resp.text = _json.dumps({
                    "error": {"code": 404, "message":
                              f"models/{sent_model} is not found for API version v1beta"}
                })
                raise _requests.HTTPError(response=resp)

            with mock.patch.object(_requests, "post", side_effect=fake_post), \
                 mock.patch("core.llm.time.sleep") as m_sleep:
                out = c.chat("sys", "user")
            self.assertIsNone(out)
            self.assertIsNotNone(c.last_error)
            self.assertIn("404", c.last_error)
            # 폴백 후보(gemini-3.5-flash-lite)가 기본 모델과 같아 추가 폴백은 없음
            # → 남은 일반 재시도 1회에서 sleep
            self.assertEqual(m_sleep.call_count, 1)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_explicit_base_url_and_model_win_over_gemini_defaults(self):
        c = self._client_with_env(
            GEMINI_API_KEY="AIzaSyTest123",
            LLM_BASE_URL="https://custom.example/v1/",
            LLM_MODEL="gemini-3.5-flash-lite",
        )
        self.assertTrue(c.is_gemini)
        self.assertEqual(c.base_url, "https://custom.example/v1")
        self.assertEqual(c.model, "gemini-3.5-flash-lite")

    def test_openai_key_keeps_existing_defaults(self):
        c = self._client_with_env(LLM_API_KEY="sk-openai-test")
        self.assertFalse(c.is_gemini)
        self.assertEqual(c.api_key, "sk-openai-test")
        self.assertEqual(c.base_url, "https://api.openai.com/v1")
        self.assertEqual(c.model, "gpt-4o-mini")

    def test_gemini_key_takes_precedence_over_llm_key(self):
        c = self._client_with_env(GEMINI_API_KEY="AIzaSyGemini", LLM_API_KEY="sk-openai")
        self.assertTrue(c.is_gemini)
        self.assertEqual(c.api_key, "AIzaSyGemini")

    def test_openai_key_wins_when_gemini_key_absent(self):
        c = self._client_with_env(LLM_API_KEY="sk-openai", OPENAI_API_KEY="sk-other")
        self.assertFalse(c.is_gemini)
        self.assertEqual(c.api_key, "sk-openai")


class TestAgentChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_data()
        cls.emit = staticmethod(lambda e: None)

    def _run_chain(self):
        notes = [
            FundamentalAgent().run(self.data, self.emit),
            TechnicalAgent().run(self.data, self.emit),
            MarketResearchAgent().run(self.data, self.emit),
        ]
        pm = PortfolioManagerAgent().run(self.data, notes, self.emit)
        return notes + [pm]

    def test_scores_and_fields(self):
        verdicts = {"적극 매수", "매수", "분할 매수", "중립 · 보유", "관망 · 감축", "매도 · 회피"}
        for note in self._run_chain():
            self.assertGreaterEqual(note["score"], 0)
            self.assertLessEqual(note["score"], 100)
            self.assertTrue(note["summary"])
            self.assertTrue(note["bullets"])
            self.assertIn(note["source"], ("rule", "llm"))
            for m in note["metrics"]:
                self.assertIn("label", m)
        pm = self._run_chain()[-1]
        self.assertIn(pm["stance"], verdicts)
        dec = pm["decision"]
        for key in ("verdict", "action", "position", "entry", "stop", "target1", "target2", "horizon", "risk_level"):
            self.assertIn(key, dec)
            self.assertTrue(str(dec[key]).strip())

    def test_deterministic(self):
        a = self._run_chain()
        b = self._run_chain()
        self.assertEqual([n["score"] for n in a], [n["score"] for n in b])

    def test_stop_below_price(self):
        dec = self._run_chain()[-1]["decision"]
        price = self.data.last_close
        stop_val = float(dec["stop"].split("(")[0].replace("$", "").replace(",", "").strip())
        self.assertLess(stop_val, price)


class TestSSEKeepaliveAndFastModel(unittest.TestCase):
    """SSE 킵얼라이브(Render/Cloudflare 100초 유휴 타임아웃 방지)와
    고속 기본 모델(gemini-3.5-flash-lite) 전환에 대한 오프라인 테스트."""

    def test_default_gemini_timeout_is_25s(self):
        import os
        from core.llm import LLMClient
        saved = os.environ.pop("LLM_TIMEOUT", None)
        saved_model = os.environ.pop("LLM_MODEL", None)
        try:
            c = LLMClient()
            self.assertEqual(c.timeout, 25.0)
        finally:
            if saved is not None:
                os.environ["LLM_TIMEOUT"] = saved
            if saved_model is not None:
                os.environ["LLM_MODEL"] = saved_model

    def test_thinking_model_bumps_max_tokens_and_sets_low_effort(self):
        """gemini-3.8-flash 명시 사용 시 Thinking 토큰 잠식 방지 안전장치."""
        import os
        from unittest import mock
        from core.llm import LLMClient
        saved = {k: os.environ.pop(k, None)
                 for k in ("GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_MODEL")}
        try:
            os.environ["GEMINI_API_KEY"] = "AIzaSyTest123"
            os.environ["LLM_MODEL"] = "gemini-3.8-flash"
            c = LLMClient()
            captured = {}

            class FakeResp:
                status_code = 200
                def raise_for_status(self): pass
                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}]}

            def fake_post(url, json=None, headers=None, timeout=None):
                captured["payload"] = json
                return FakeResp()

            with mock.patch("core.llm.requests.post", side_effect=fake_post):
                out = c.chat("sys", "user", max_tokens=700)  # 기존 기본값(700) 그대로 호출
            self.assertEqual(out, "ok")
            self.assertGreaterEqual(captured["payload"]["max_tokens"], 4096)
            self.assertEqual(captured["payload"]["reasoning_effort"], "low")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_fast_lite_model_keeps_original_max_tokens(self):
        """기본 경량 모델(3.5-flash-lite)은 호출자의 max_tokens 를 그대로 존중."""
        import os
        from unittest import mock
        from core.llm import LLMClient
        saved = {k: os.environ.pop(k, None)
                 for k in ("GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_MODEL")}
        try:
            os.environ["GEMINI_API_KEY"] = "AIzaSyTest123"
            c = LLMClient()
            captured = {}

            class FakeResp:
                status_code = 200
                def raise_for_status(self): pass
                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}]}

            def fake_post(url, json=None, headers=None, timeout=None):
                captured["payload"] = json
                return FakeResp()

            with mock.patch("core.llm.requests.post", side_effect=fake_post):
                c.chat("sys", "user", max_tokens=700)
            self.assertEqual(captured["payload"]["max_tokens"], 700)
            self.assertNotIn("reasoning_effort", captured["payload"])
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_retry_is_two_attempts_with_short_wait(self):
        """재시도 총 2회(attempt in range(2)), 대기 1.0초 — 최악에도 빠른 폴백."""
        import os
        from unittest import mock
        from core.llm import LLMClient
        saved = {k: os.environ.pop(k, None)
                 for k in ("GEMINI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_MODEL")}
        try:
            os.environ["LLM_API_KEY"] = "sk-test"
            c = LLMClient()
            err = ConnectionError("boom")
            with mock.patch("core.llm.requests.post", side_effect=err) as m_post, \
                 mock.patch("core.llm.time.sleep") as m_sleep:
                out = c.chat("sys", "user")
            self.assertIsNone(out)
            self.assertEqual(m_post.call_count, 2)   # 시도 총 2회
            self.assertEqual(m_sleep.call_count, 1)  # 시도 사이 1회 대기
            m_sleep.assert_called_once_with(1.0)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_sse_stream_sends_keepalive_and_headers(self):
        """큐가 비어도 keepalive 주석을 흘려보내고, 프록시 방지 헤더를 보낸다."""
        import asyncio
        import queue as queue_mod
        from types import SimpleNamespace
        from unittest import mock
        import app as app_module
        from fastapi.responses import StreamingResponse

        class KeepaliveTestQueue(queue_mod.Queue):
            """첫 get 은 타임아웃(Empty) → keepalive 경로, 이후엔 정상 동작."""
            def get(self, *a, **k):
                if not getattr(self, "_keepalive_tested", False):
                    self._keepalive_tested = True
                    raise queue_mod.Empty()
                return super().get(*a, **k)

        def fake_pipeline(symbol, emit):
            emit({"type": "log", "stage": "data", "message": "테스트 로그"})
            emit({"type": "end"})

        # app.py 가 임포트한 queue 모듈 자리에 테스트용 큐 클래스만 갈아끼운다.
        fake_queue_mod = SimpleNamespace(Queue=KeepaliveTestQueue, Empty=queue_mod.Empty)
        with mock.patch("app.run_pipeline", side_effect=fake_pipeline), \
             mock.patch.object(app_module, "queue", fake_queue_mod):
            resp = app_module.analyze_stream(symbol="AAPL")
            self.assertIsInstance(resp, StreamingResponse)
            self.assertEqual(resp.headers.get("x-accel-buffering"), "no")
            self.assertIn("no-transform", resp.headers.get("cache-control", ""))

            # TestClient/httpx 의존성 없이 body_iterator 를 직접 소비한다.
            async def _collect() -> str:
                chunks = []
                async for chunk in resp.body_iterator:
                    chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
                return "".join(chunks)

            body = asyncio.run(_collect())
        self.assertIn(": keepalive\n\n", body)
        self.assertIn("event: log", body)
        self.assertIn("테스트 로그", body)


if __name__ == "__main__":
    unittest.main()
