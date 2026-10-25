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

    def test_llm_disabled_falls_back_to_rule(self):
        import os
        os.environ.pop("LLM_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        nar, source = _llm_note("t", "역할", 50.0, "중립", "facts", lambda e: None)
        self.assertIsNone(nar)
        self.assertEqual(source, "rule")


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


if __name__ == "__main__":
    unittest.main()
