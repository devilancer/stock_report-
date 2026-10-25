"""Four analyst agents + the research pipeline.

에이전트 체인:
  01 펀더멘털 애널리스트  — 가치/수익성/성장/재무건전성
  02 테크니컬 애널리스트  — 추세/모멘텀/변동성/수급
  03 시장 리서치 애널리스트 — 시장 레짐/상대강도/거시 환경
  04 포트폴리오 매니저    — 3개 노트를 종합해 최종 결정

각 에이전트는 동일한 `AgentReport` 스키마로 결과를 출력하며,
포트폴리오 매니저는 앞선 3개 노트를 입력으로 사용한다.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import indicators as ta
from .data import Emit, StockData, DataCollector, DataError
from .llm import get_llm

EmitFn = Callable[[dict], None]

# ------------------------------------------------------------------ helpers


def _f(v, digits: int = 2):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if math.isfinite(f) else None


def fmt_pct(v, digits: int = 1) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return f"{f * 100:.{digits}f}%"


def fmt_num(v, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return f"{f:,.{digits}f}"


def fmt_cap(v, currency: str) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f) or f <= 0:
        return "—"
    if currency == "KRW":
        if f >= 1e12:
            return f"{f / 1e12:,.2f}조원"
        return f"{f / 1e8:,.0f}억원"
    if f >= 1e12:
        return f"{f / 1e12:,.2f}T {currency}"
    if f >= 1e9:
        return f"{f / 1e9:,.2f}B {currency}"
    return f"{f / 1e6:,.0f}M {currency}"


def fmt_price(v, currency: str) -> str:
    if v is None:
        return "—"
    unit = {"KRW": "원", "USD": "$", "CNY": "¥"}.get(currency, "")
    if currency == "USD":
        return f"${v:,.2f}"
    return f"{v:,.0f}{unit}"


def _tone(score: float, good=65, bad=45) -> str:
    return "good" if score >= good else ("bad" if score < bad else "neutral")


class Score:
    """Weighted score accumulator with 0-100 clipping."""

    def __init__(self):
        self.total = 0.0
        self.weight = 0.0

    def add(self, value: float, weight: float):
        if value is None or not math.isfinite(value):
            return
        self.total += value * weight
        self.weight += weight

    @property
    def value(self) -> float:
        if self.weight == 0:
            return 50.0
        return max(0.0, min(100.0, self.total / self.weight))


def stance_label(score: float) -> str:
    if score >= 70:
        return "긍정"
    if score >= 58:
        return "약간 긍정"
    if score >= 45:
        return "중립"
    if score >= 32:
        return "부정적"
    return "매우 부정"


# ------------------------------------------------------------------ LLM narrative

_LLM_SYSTEM = (
    "당신은 한국어 투자 리서치 노트를 작성하는 베테랑 애널리스트입니다.\n"
    "규칙:\n"
    "- 출력은 반드시 JSON 객체 하나만: {\"summary\": \"...\", \"bullets\": [\"...\"], \"risks\": [\"...\"]}\n"
    "- summary: 2~3문장 요약 / bullets: 3~5개(각 1문장 판단 근거) / risks: 0~3개(경고)\n"
    "- 반드시 제공된 수치만 인용하고, 수치를 지어내지 않습니다.\n"
    "- 투자 권유·예언투 표현 금지. '~로 판단됩니다', '~가 확인됩니다' 서술형.\n"
    "- 간결한 존댓말 전문가 어조."
)


def _facts_block(metrics: list[dict], bullets: list[str], risks: list[str]) -> str:
    lines = ["[지표]"]
    for m in metrics:
        lines.append(f"- {m['label']}: {m['value']} ({m.get('note', '')})")
    if bullets:
        lines.append("[규칙 기반 1차 판단]")
        lines += [f"- {b}" for b in bullets]
    if risks:
        lines.append("[식별된 리스크]")
        lines += [f"- {r}" for r in risks]
    return "\n".join(lines)


def _llm_note(key: str, role: str, score: float, stance: str, facts: str,
              emit: EmitFn) -> tuple[dict | None, str]:
    """LLM 코멘티 생성. 실패/미설정 시 (None, 'rule') 반환."""
    llm = get_llm()
    if not llm.enabled:
        return None, "rule"
    emit({"type": "log", "stage": key, "message": f"LLM 코멘티 생성 중 ({llm.model})…"})
    user = f"역할: {role}\n종합 점수: {score}/100 · 스탠스: {stance}\n\n분석 팩트:\n{facts}"
    out = llm.complete_json(_LLM_SYSTEM, user)
    if not out or not isinstance(out.get("summary"), str) or not out["summary"].strip():
        if llm.last_error:
            emit({"type": "log", "stage": key,
                  "message": f"LLM 호출 실패 ({llm.last_error}) → 규칙 기반 폴백"})
        else:
            emit({"type": "log", "stage": key,
                  "message": "LLM 응답 JSON 파싱 실패 → 규칙 기반 폴백"})
        return None, "rule"
    # 키는 소문자로 정규화되어 돌아오지만, 대소문자 혼용 대비 안전 조회
    summary = out.get("summary") or out.get("Summary") or ""
    bullets = [str(b).strip() for b in (out.get("bullets") or out.get("Bullets") or []) if str(b).strip()][:5]
    risks = [str(r).strip() for r in (out.get("risks") or out.get("Risks") or []) if str(r).strip()][:3]
    emit({"type": "log", "stage": key, "message": f"LLM 코멘티 완성 ({llm.model})"})
    return {"summary": summary.strip(), "bullets": bullets, "risks": risks}, "llm"


# ------------------------------------------------------------------ agents


class FundamentalAgent:
    key = "fundamental"
    index = 1
    title = "펀더멘털 애널리스트"
    subtitle = "가치 · 수익성 · 성장 · 재무건전성"

    def run(self, data: StockData, emit: EmitFn) -> dict[str, Any]:
        emit({"type": "log", "stage": self.key, "message": "밸류에이션·수익성·성장·재무 구조 분석 시작"})
        p = data.profile
        metrics: list[dict[str, Any]] = []
        bullets: list[str] = []
        risks: list[str] = []
        sc = Score()

        # ---- valuation
        pe = p.get("pe_ttm") or p.get("pe_forward")
        if pe is not None and pe > 0:
            v = 78 if pe < 12 else 66 if pe < 20 else 52 if pe < 30 else 38 if pe < 45 else 25
            note = "실적 대비 저평가 구간" if v >= 66 else ("성장 프리미엄이 낮은 구간" if v >= 52 else "성장 기대감이 가격에 선반영")
            metrics.append({"label": "PER (TTM)" if p.get("pe_ttm") else "PER (Fwd)", "value": f"{pe:,.1f}배", "tone": _tone(v), "note": note})
            sc.add(v, 24)
            bullets.append(f"현재 PER은 {pe:,.1f}배로, 밸류에이션 점수는 {v}점입니다.")
        elif pe is not None and pe <= 0:
            metrics.append({"label": "PER (TTM)", "value": "적자", "tone": "bad", "note": "당기순손실 구간"})
            sc.add(28, 24)
            bullets.append("최근 실적이 적자 구간이어서 PER 기반 밸류에이션이 무의미합니다.")
        else:
            bullets.append("PER 데이터가 제공되지 않아 밸류에이션 판단을 제외했습니다.")

        pb = p.get("pb")
        if pb is not None and pb > 0:
            v = 75 if pb < 1.2 else 62 if pb < 2.5 else 48 if pb < 5 else 32
            metrics.append({"label": "PBR", "value": f"{pb:,.2f}배", "tone": _tone(v), "note": "자산 대비 낮은 프리미엄" if v >= 62 else "자산 대비 높은 프리미엄"})
            sc.add(v, 12)
        # ---- profitability
        roe = p.get("roe")
        if roe is not None:
            v = 82 if roe >= 0.20 else 68 if roe >= 0.12 else 52 if roe >= 0.06 else 34
            metrics.append({"label": "ROE", "value": fmt_pct(roe), "tone": _tone(v), "note": "자본수익성 양호" if v >= 68 else ("자본수익성 보통" if v >= 52 else "자본수익성 취약")})
            sc.add(v, 20)
            bullets.append(f"ROE는 {fmt_pct(roe)}로 자본 대비 {'우수한' if v >= 68 else '보통 수준의'} 이익 창출력을 보여줍니다.")
        nm = p.get("net_margin")
        if nm is not None:
            v = 80 if nm >= 0.20 else 66 if nm >= 0.10 else 50 if nm >= 0.03 else 30
            metrics.append({"label": "순이익률", "value": fmt_pct(nm), "tone": _tone(v), "note": "마진 구조가 두텁습니다" if v >= 66 else "마진이 얇은 사업 구조"})
            sc.add(v, 10)
        # ---- growth
        rg = p.get("revenue_growth")
        if rg is not None:
            v = 80 if rg >= 0.25 else 66 if rg >= 0.10 else 52 if rg >= 0.0 else 32
            metrics.append({"label": "매출 성장률", "value": fmt_pct(rg), "tone": _tone(v), "note": "성장 궤도" if v >= 66 else ("성장 정체" if v >= 52 else "매출 역성장")})
            sc.add(v, 16)
            bullets.append(f"매출은 전년 대비 {fmt_pct(abs(rg))} {'증가' if rg >= 0 else '감소'}했습니다.")
            if v < 52:
                risks.append("매출 성장이 정체 또는 역성장 중입니다.")
        eg = p.get("earnings_growth")
        if eg is not None:
            v = 80 if eg >= 0.25 else 64 if eg >= 0.08 else 50 if eg >= -0.05 else 30
            metrics.append({"label": "이익 성장률", "value": fmt_pct(eg), "tone": _tone(v), "note": "이익 개선세" if v >= 64 else "이익 둔화 신호"})
            sc.add(v, 12)
        # ---- balance sheet
        de = p.get("debt_to_equity")
        if de is not None:
            v = 78 if de < 40 else 64 if de < 90 else 48 if de < 160 else 30
            metrics.append({"label": "부채비율", "value": f"{de:,.0f}%", "tone": _tone(v), "note": "재무 레버리지 낮음" if v >= 64 else "차입 의존도 확인 필요"})
            sc.add(v, 12)
            if v < 48:
                risks.append("부채비율이 높아 금리·유동성 환경 변화에 민감합니다.")
        # ---- dividend
        dy = p.get("dividend_yield")
        if dy is not None:
            dv = dy / 100  # yfinance dividendYield는 % 단위(예: 0.31 = 0.31%)
            v = 72 if dv >= 0.03 else 60 if dv >= 0.015 else 50
            metrics.append({"label": "배당수익률", "value": fmt_pct(dv), "tone": _tone(v), "note": "현금흐름 환원 존재" if dv >= 0.015 else "배당 매력 제한적"})
            sc.add(v, 6)

        score = _f(sc.value, 1)
        stance = stance_label(score)
        summary = self._summary(data, score, pe, roe, rg)
        nar, source = _llm_note(self.key, f"{self.title} ({self.subtitle})", score, stance,
                                _facts_block(metrics, bullets, risks), emit)
        if nar:
            summary = nar["summary"]
            bullets = nar["bullets"] or bullets
            risks = nar["risks"] or risks
        emit({"type": "log", "stage": self.key, "message": f"펀더멘털 점수 {score:.1f}점 · {stance}"})
        return {
            "key": self.key, "index": self.index, "title": self.title, "subtitle": self.subtitle,
            "score": score, "stance": stance, "confidence": _f(min(1.0, data.coverage + 0.25), 2),
            "source": source,
            "summary": summary, "metrics": metrics, "bullets": bullets,
            "risks": risks or ["펀더멘털 기반 특이 리스크는 제한적입니다."],
        }

    def _summary(self, data: StockData, score: float, pe, roe, rg) -> str:
        name = data.name
        parts = [f"{name}의 펀더멘털을 종합하면 {score:.0f}점({stance_label(score)}) 수준입니다."]
        if pe is not None and pe > 0:
            parts.append(f"현재 PER {pe:,.1f}배는")
            parts.append("이익 대비 합리적~저평가 구간에 위치합니다." if pe < 22 else "성장 기대감이 상당 부분 반영된 구간입니다.")
        if roe is not None:
            parts.append(f"ROE {fmt_pct(roe)}로 수익성은 {'견조합니다' if roe >= 0.12 else '개선이 필요한 상태입니다'}.")
        if rg is not None:
            parts.append(f"매출 성장률은 {fmt_pct(rg)}입니다.")
        return " ".join(parts)


class TechnicalAgent:
    key = "technical"
    index = 2
    title = "테크니컬 애널리스트"
    subtitle = "추세 · 모멘텀 · 변동성 · 수급"

    def run(self, data: StockData, emit: EmitFn) -> dict[str, Any]:
        emit({"type": "log", "stage": self.key, "message": "추세·모멘텀·변동성 지표 계산 시작"})
        df = data.prices
        close = df["close"]
        last = float(close.iloc[-1])
        metrics: list[dict[str, Any]] = []
        bullets: list[str] = []
        risks: list[str] = []
        sc = Score()

        sma20 = float(df["sma20"].iloc[-1]) if pd.notna(df["sma20"].iloc[-1]) else None
        sma60 = float(df["sma60"].iloc[-1]) if pd.notna(df["sma60"].iloc[-1]) else None
        sma120 = float(df["sma120"].iloc[-1]) if pd.notna(df["sma120"].iloc[-1]) else None

        # ---- trend
        trend_pts = 50.0
        above = sum(1 for s in (sma20, sma60, sma120) if s and last > s)
        trend_pts += (above - 1.5) * 12
        if sma20 and sma60:
            cross = "골든크로스" if sma20 > sma60 else "데드크로스"
            trend_pts += 8 if sma20 > sma60 else -8
            bullets.append(f"단기 이평선(SMA20)이 중기(SMA60) {cross} 상태입니다.")
        metrics.append({
            "label": "추세 정렬", "value": f"{above}/3 상방",
            "tone": _tone(trend_pts), "note": "이동평균선 상대 위치",
        })
        sc.add(trend_pts, 30)

        # ---- momentum
        rsi_v = float(df["rsi14"].iloc[-1]) if pd.notna(df["rsi14"].iloc[-1]) else None
        if rsi_v is not None:
            r_pts = 62 if 50 <= rsi_v <= 65 else 55 if 45 <= rsi_v < 70 else 40 if rsi_v >= 70 else 45
            if rsi_v < 30:
                r_pts = 58  # 과매도 → 되돌림 기대
            metrics.append({"label": "RSI (14)", "value": f"{rsi_v:.1f}", "tone": _tone(r_pts), "note": "과매수 경계" if rsi_v >= 70 else ("과매도 반등 구간" if rsi_v < 30 else "중립 모멘텀")})
            bullets.append(f"RSI(14)는 {rsi_v:.1f}로 {'과매수' if rsi_v >= 70 else '과매도' if rsi_v < 30 else '중립'} 영역입니다.")
            sc.add(r_pts, 18)

        hist = df["macd_hist"]
        if len(hist.dropna()) > 3:
            h = float(hist.iloc[-1])
            h_prev = float(hist.iloc[-2])
            m_pts = 65 if h > 0 and h > h_prev else 58 if h > 0 else 38 if h < h_prev else 45
            metrics.append({"label": "MACD 히스토그램", "value": f"{h:,.1f}", "tone": _tone(m_pts), "note": "모멘텀 확대" if h > h_prev else "모멘텀 둔화"})
            sc.add(m_pts, 18)

        r3m = ta.pct_return(close, 63)
        r1m = ta.pct_return(close, 21)
        for label, r, w in (("3개월 수익률", r3m, 12), ("1개월 수익률", r1m, 6)):
            if r is not None and math.isfinite(r):
                p_ = 68 if r >= 0.12 else 58 if r >= 0.03 else 44 if r >= -0.08 else 32
                metrics.append({"label": label, "value": fmt_pct(r), "tone": _tone(p_), "note": "상승 추세 지속" if p_ >= 58 else "하락 압력"})
                sc.add(p_, w)

        # ---- volatility / range
        atr_v = float(df["atr14"].iloc[-1]) if pd.notna(df["atr14"].iloc[-1]) else None
        vol = ta.annualized_vol(close, 30)
        if vol is not None and math.isfinite(vol):
            metrics.append({"label": "연율화 변동성", "value": fmt_pct(vol, 0), "tone": "neutral", "note": "최근 30일 기준"})
        if atr_v:
            metrics.append({"label": "ATR (14)", "value": fmt_price(atr_v, data.currency), "tone": "neutral", "note": f"일일 진폭 {fmt_pct(atr_v / last)} 수준"})

        hi, lo = data.profile.get("week52_high"), data.profile.get("week52_low")
        if hi and lo and hi > lo:
            pos = (last - lo) / (hi - lo) * 100
            rng_pts = 70 if pos >= 80 else 60 if pos >= 55 else 50 if pos >= 35 else 40
            metrics.append({"label": "52주 레인지 포지션", "value": f"{pos:.0f}%", "tone": _tone(rng_pts), "note": f"고가 {fmt_price(hi, data.currency)} / 저가 {fmt_price(lo, data.currency)}"})
            sc.add(rng_pts, 12)
            bullets.append(f"현재가는 52주 범위의 {pos:.0f}% 지점에 위치합니다.")

        vol_ratio = float(df["volume"].iloc[-20:].mean() / max(df["volume"].iloc[-60:].mean(), 1))
        if math.isfinite(vol_ratio):
            v_pts = 62 if vol_ratio > 1.15 else 52 if vol_ratio > 0.8 else 42
            metrics.append({"label": "거래량 추이", "value": f"{vol_ratio:,.2f}x", "tone": _tone(v_pts), "note": "20일/60일 평균 대비"})
            sc.add(v_pts, 10)

        score = _f(sc.value, 1)
        stance = stance_label(score)
        summary = (
            f"{data.name}의 차트는 {stance} 상태({score:.0f}점)입니다. "
            f"현재가 {fmt_price(last, data.currency)}는 SMA20 {'위' if sma20 and last > sma20 else '아래'}, "
            f"SMA60 {'위' if sma60 and last > sma60 else '아래'}에 위치하며, "
            f"3개월 모멘텀은 {fmt_pct(r3m)}입니다."
        )
        if vol and math.isfinite(vol) and vol > 0.45:
            risks.append(f"연율화 변동성 {fmt_pct(vol, 0)}로 높은 편 — 포지션 크기 축소가 필요합니다.")
        if rsi_v is not None and rsi_v >= 72:
            risks.append("RSI 과매수 구간 — 단기 조정 가능성이 있습니다.")
        if rsi_v is not None and rsi_v <= 28:
            risks.append("RSI 과매도 구간 — 반등 시도 전 추가 하락 여지가 있습니다.")
        if sma20 and sma60 and sma20 < sma60:
            risks.append("단기 이평선이 중기 이평선 아래에 있어 하방 추세가 우세합니다.")

        nar, source = _llm_note(self.key, f"{self.title} ({self.subtitle})", score, stance,
                                _facts_block(metrics, bullets, risks), emit)
        if nar:
            summary = nar["summary"]
            bullets = nar["bullets"] or bullets
            risks = nar["risks"] or risks
        emit({"type": "log", "stage": self.key, "message": f"테크니컬 점수 {score:.1f}점 · {stance}"})
        return {
            "key": self.key, "index": self.index, "title": self.title, "subtitle": self.subtitle,
            "score": score, "stance": stance, "confidence": 0.8, "source": source,
            "summary": summary, "metrics": metrics, "bullets": bullets,
            "risks": risks or ["차트 기반 특이 리스크는 제한적입니다."],
        }


class MarketResearchAgent:
    key = "market"
    index = 3
    title = "시장 리서치 애널리스트"
    subtitle = "시장 레짐 · 상대강도 · 거시 환경"

    def run(self, data: StockData, emit: EmitFn) -> dict[str, Any]:
        emit({"type": "log", "stage": self.key, "message": f"{data.index_name} 레짐 분석 및 상대강도 측정"})
        metrics: list[dict[str, Any]] = []
        bullets: list[str] = []
        risks: list[str] = []
        sc = Score()

        idx = data.index_prices
        if idx is not None and len(idx) >= 60:
            iclose = idx["close"]
            last_i = float(iclose.iloc[-1])
            sma50 = ta.sma(iclose, 50)
            sma200 = ta.sma(iclose, 200)
            above50 = last_i > float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
            reg = 62.0
            if above50:
                reg += 12
            else:
                reg -= 12
            if pd.notna(sma200.iloc[-1]):
                reg += 10 if last_i > float(sma200.iloc[-1]) else -10
            r3m_i = ta.pct_return(iclose, 63)
            if r3m_i is not None and math.isfinite(r3m_i):
                reg += 10 if r3m_i >= 0.05 else 0 if r3m_i >= -0.03 else -12
            reg = max(5.0, min(95.0, reg))
            sc.add(reg, 34)
            metrics.append({"label": f"{data.index_name} 추세", "value": f"{last_i:,.0f}", "tone": _tone(reg), "note": "SMA50 " + ("위" if above50 else "아래")})
            metrics.append({"label": f"{data.index_name} 3M", "value": fmt_pct(r3m_i), "tone": _tone(reg), "note": "지수 모멘텀"})
            bullets.append(
                f"{data.index_name} 지수는 SMA50 {'상회' if above50 else '하회'}, "
                f"3개월 모멘텀 {fmt_pct(r3m_i)}로 {'위험선호(Risk-On)' if reg >= 65 else '중립' if reg >= 45 else '위험회피(Risk-Off)'} 환경입니다."
            )
            if reg < 45:
                risks.append("시장 전체가 하방 추세 — 개별 종목 상승 탄력이 제한될 수 있습니다.")
        else:
            sc.add(50, 20)
            bullets.append("시장 지수 데이터가 부족해 레짐 판단이 제한적입니다.")
            risks.append("시장 컨텍스트 데이터 부족으로 판단 신뢰도가 낮습니다.")

        # volatility gauge (US: VIX)
        if data.vol_index is not None and len(data.vol_index) >= 20:
            vix = float(data.vol_index["close"].iloc[-1])
            v_pts = 68 if vix < 17 else 55 if vix < 23 else 38 if vix < 30 else 25
            metrics.append({"label": "공포지수(VIX)", "value": f"{vix:,.1f}", "tone": _tone(v_pts), "note": "옵션 시장 내재변동성"})
            sc.add(v_pts, 16)
            bullets.append(f"VIX는 {vix:,.1f}로 {'위험선호' if vix < 17 else '경계' if vix < 23 else '공포'} 영역입니다.")

        # relative strength vs index
        if idx is not None and len(idx) >= 63:
            r3m_s = ta.pct_return(data.prices["close"], 63)
            r3m_i = ta.pct_return(idx["close"], 63)
            if r3m_s is not None and r3m_i is not None and math.isfinite(r3m_s) and math.isfinite(r3m_i):
                rs = r3m_s - r3m_i
                rs_pts = 72 if rs >= 0.08 else 58 if rs >= 0 else 42 if rs >= -0.08 else 30
                sc.add(rs_pts, 26)
                metrics.append({"label": "상대강도(vs 지수)", "value": fmt_pct(rs), "tone": _tone(rs_pts), "note": "3개월 초과수익률"})
                bullets.append(f"지수 대비 3개월 초과수익률은 {fmt_pct(rs)}로 아웃퍼폼" if rs >= 0 else f"지수 대비 3개월 초과수익률 {fmt_pct(rs)}로 언더퍼폼 중입니다.")

        # stock volatility regime
        sv = ta.annualized_vol(data.prices["close"], 30)
        if sv is not None and math.isfinite(sv):
            sv_pts = 60 if sv < 0.28 else 50 if sv < 0.45 else 36
            sc.add(sv_pts, 24)
            metrics.append({"label": "개별 종목 변동성", "value": fmt_pct(sv, 0), "tone": _tone(sv_pts), "note": "연율화 30일"})
            if sv >= 0.45:
                risks.append("종목 변동성이 시장 평균 대비 높아 헤지/분할매수 전략이 유효합니다.")

        score = _f(sc.value, 1)
        regime = "위험선호(Risk-On)" if score >= 65 else "중립(Neutral)" if score >= 45 else "위험회피(Risk-Off)"
        summary = (
            f"현재 시장 레짐은 {regime}로 진단됩니다({score:.0f}점). "
            f"{data.name}은(는) 시장 지수 대비 상대적으로 "
            f"{'강한' if score >= 65 else '보통' if score >= 45 else '약한'} 흐름을 보이고 있습니다."
        )
        nar, source = _llm_note(self.key, f"{self.title} ({self.subtitle})", score, regime,
                                _facts_block(metrics, bullets, risks), emit)
        if nar:
            summary = nar["summary"]
            bullets = nar["bullets"] or bullets
            risks = nar["risks"] or risks
        emit({"type": "log", "stage": self.key, "message": f"시장 리서치 점수 {score:.1f}점 · {regime}"})
        return {
            "key": self.key, "index": self.index, "title": self.title, "subtitle": self.subtitle,
            "score": score, "stance": regime, "confidence": 0.75, "source": source,
            "summary": summary, "metrics": metrics, "bullets": bullets,
            "risks": risks or ["거시 환경 기반 특이 리스크는 제한적입니다."],
        }


class PortfolioManagerAgent:
    key = "portfolio"
    index = 4
    title = "포트폴리오 매니저"
    subtitle = "종합 판단 · 포지션 설계 · 리스크 관리"

    def run(self, data: StockData, notes: list[dict[str, Any]], emit: EmitFn) -> dict[str, Any]:
        emit({"type": "log", "stage": self.key, "message": "3개 리서치 노트를 종합해 최종 결정 생성"})
        fund = next(n for n in notes if n["key"] == "fundamental")
        tech = next(n for n in notes if n["key"] == "technical")
        mkt = next(n for n in notes if n["key"] == "market")

        coverage_bonus = (data.coverage - 0.6) * 12
        final = 0.36 * fund["score"] + 0.29 * tech["score"] + 0.25 * mkt["score"] + 10 + coverage_bonus
        final = max(2.0, min(98.0, final))
        score = _f(final, 1)

        scores = np.array([fund["score"], tech["score"], mkt["score"]], dtype=float)
        agreement = 1.0 - min(1.0, float(scores.std()) / 22.0)
        confidence = _f(min(0.95, 0.35 + 0.4 * agreement + 0.25 * data.coverage), 2)

        verdict, tier = self._verdict(final)
        decision = self._plan(data, final, verdict, tier, tech["score"], mkt["score"], confidence)
        metrics = [
            {"label": "종합 점수", "value": f"{score:.1f} / 100", "tone": _tone(final, 60, 45), "note": "3개 노트 가중 종합"},
            {"label": "펀더멘털", "value": f"{fund['score']:.1f}", "tone": _tone(fund["score"]), "note": f"비중 36% · {fund['stance']}"},
            {"label": "테크니컬", "value": f"{tech['score']:.1f}", "tone": _tone(tech["score"]), "note": f"비중 29% · {tech['stance']}"},
            {"label": "시장 환경", "value": f"{mkt['score']:.1f}", "tone": _tone(mkt["score"]), "note": f"비중 25% · {mkt['stance']}"},
            {"label": "에이전트 합의도", "value": f"{agreement * 100:.0f}%", "tone": "neutral", "note": "3개 노트 점수 일치 정도"},
            {"label": "데이터 커버리지", "value": f"{data.coverage * 100:.0f}%", "tone": "neutral", "note": "확보한 펀더멘털 지표 비율"},
        ]
        bullets = [
            f"가중 종합 점수는 {score:.1f}점이며, 이는 '{verdict}' 판단으로 이어집니다.",
            f"펀더멘털 {fund['score']:.0f}점 · 테크니컬 {tech['score']:.0f}점 · 시장 {mkt['score']:.0f}점이 "
            f"{'방향을 함께 가리키고 있어' if agreement >= 0.75 else '일부 엇갈려'} 판단 신뢰도는 {confidence * 100:.0f}%입니다.",
        ]
        if tech["score"] < 45 and fund["score"] >= 60:
            bullets.append("펀더멘털은 견조하나 차트가 약합니다 — 분할 진입으로 타이밍 리스크를 흡수하는 전략이 적합합니다.")
        if mkt["score"] < 45:
            bullets.append("시장 레짐이 부정적입니다 — 신규 자금 투입 속도를 늦추고 시장 반전 신호를 확인하세요.")
        if fund["score"] < 45 and tech["score"] >= 60:
            bullets.append("차트는 강하나 내재가치 지지가 약합니다 — 추세 이탈 시 신속한 손절이 전제되어야 합니다.")

        risks: list[str] = []
        for n in notes:
            for r in n.get("risks", [])[:2]:
                risks.append(f"[{n['title'].split(' ')[0]}] {r}")

        summary = (
            f"3개 에이전트 노트를 종합한 최종 판단은 '{verdict}'입니다. "
            f"종합 점수 {score:.1f}점, 판단 신뢰도 {confidence * 100:.0f}%이며, "
            f"권고 포지션은 포트폴리오의 {decision['position']}입니다."
        )
        facts = self._facts(notes, decision, data)
        nar, source = _llm_note(self.key, f"{self.title} ({self.subtitle})", score, verdict,
                                facts, emit)
        if nar:
            summary = nar["summary"]
            bullets = nar["bullets"] or bullets
        emit({"type": "log", "stage": self.key, "message": f"최종 결정: {verdict} (종합 {score:.1f}점)"})
        return {
            "key": self.key, "index": self.index, "title": self.title, "subtitle": self.subtitle,
            "score": score, "stance": verdict, "confidence": confidence, "source": source,
            "summary": summary, "metrics": metrics, "bullets": bullets,
            "risks": risks[:6], "decision": decision,
        }

    def _facts(self, notes: list[dict[str, Any]], decision: dict[str, Any], data: StockData) -> str:
        lines = [f"[종목] {data.name} ({data.yahoo_symbol}) · 최근 종가 {fmt_price(data.last_close, data.currency)}"]
        for n in notes:
            lines.append(f"[{n['title']}] 점수 {n['score']:.1f} · {n['stance']}")
            lines.append(f"  요약: {n['summary']}")
        lines.append("[결정론적 최종 결정 필드 — 이 값은 변경하지 않습니다]")
        for k, v in decision.items():
            if k == "notes":
                continue
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _verdict(self, score: float):
        if score >= 72:
            return "적극 매수", "strong_buy"
        if score >= 60:
            return "매수", "buy"
        if score >= 52:
            return "분할 매수", "accumulate"
        if score >= 44:
            return "중립 · 보유", "hold"
        if score >= 34:
            return "관망 · 감축", "reduce"
        return "매도 · 회피", "sell"

    def _plan(self, data: StockData, score: float, verdict: str, tier: str,
              tech_score: float, mkt_score: float, confidence: float) -> dict[str, Any]:
        price = data.last_close
        cur = data.currency
        df = data.prices
        atr_v = float(df["atr14"].iloc[-1]) if pd.notna(df["atr14"].iloc[-1]) else price * 0.02
        atr_pct = max(0.005, atr_v / price)
        stop_pct = float(min(0.25, max(0.08, 2.2 * atr_pct)))
        vol = ta.annualized_vol(df["close"], 30)
        vol = vol if vol and math.isfinite(vol) else 0.30

        if tier in ("strong_buy", "buy", "accumulate"):
            base = 0.06 if tier != "strong_buy" else 0.08
            size = base * max(0.55, min(1.5, 0.16 / vol)) * (0.7 + 0.5 * confidence)
            size = float(min(0.12, max(0.02, size)))
            position = f"{size * 100:.1f}%"
            if tier == "accumulate":
                entry = (f"1차 {size * 100 * 0.5:.1f}%p는 현재가 {fmt_price(price, cur)} 부근에서,"
                         f" 2차는 조정 시 SMA20({fmt_price(float(df['sma20'].iloc[-1]), cur)}) 부근에서 분할 진입")
            else:
                entry = (f"정가 매수 후 {fmt_price(price * 0.97, cur)}(-3%) 이하 조정 시 추가 매수로 평균 단가 관리")
            action = "신규 매수"
        elif tier == "hold":
            position = "0–3% (기존 보유분 유지)"
            entry = "신규 진입 보류 · 보유 비중 유지, 추가 매수는 섹터/시장 반전 확인 후"
            action = "보유"
        else:
            position = "0–1% (신규 매수 금지)"
            entry = "기존 포지션은 반등 시 분할 매도, 신규 자금 투입 중단"
            action = "감축 / 회피"

        stop = price * (1 - stop_pct)
        t1 = price * (1 + 1.5 * stop_pct)
        t2 = price * (1 + 2.6 * stop_pct)
        horizon = "3–12개월 (중기)" if score >= 52 else "1–3개월 (단기 대응)"

        return {
            "verdict": verdict,
            "action": action,
            "position": position,
            "entry": entry,
            "stop": f"{fmt_price(stop, cur)} ({-stop_pct * 100:.1f}%)",
            "target1": f"{fmt_price(t1, cur)} (+{1.5 * stop_pct * 100:.1f}%)",
            "target2": f"{fmt_price(t2, cur)} (+{2.6 * stop_pct * 100:.1f}%)",
            "horizon": horizon,
            "risk_level": "높음" if vol >= 0.45 else ("중간" if vol >= 0.25 else "낮음"),
            "daily_move": f"일일 평균 진폭 {fmt_price(atr_v, cur)} ({fmt_pct(atr_pct)})",
            "notes": [
                f"목표가1/2은 ATR 기반 리스크:리워드 1:1.5 / 1:2.6 구조로 설계했습니다.",
                f"52주 고가 {fmt_price(data.profile.get('week52_high'), cur)} 부근은 저항 구간으로 참고하세요.",
                f"손절 라인 {fmt_price(stop, cur)} 이탈 시 논리가 훼손되므로 즉시 감축합니다.",
            ],
        }


# ------------------------------------------------------------------ pipeline


def run_pipeline(symbol: str, emit: EmitFn) -> dict[str, Any]:
    """Run the full 4-agent chain and return the curated report."""
    collector = DataCollector()
    emit({"type": "stage", "key": "data", "status": "running"})
    data = collector.collect(symbol, emit)
    emit({"type": "stage", "key": "data", "status": "done", "meta": {
        "name": data.name, "symbol": data.yahoo_symbol, "market": data.market,
        "currency": data.currency, "price": data.last_close,
    }})

    notes: list[dict[str, Any]] = []
    for agent in (FundamentalAgent(), TechnicalAgent(), MarketResearchAgent()):
        emit({"type": "stage", "key": agent.key, "status": "running"})
        note = agent.run(data, emit)
        notes.append(note)
        emit({"type": "stage", "key": agent.key, "status": "done", "report": note})

    pm = PortfolioManagerAgent()
    emit({"type": "stage", "key": pm.key, "status": "running"})
    final = pm.run(data, notes, emit)
    notes.append(final)
    emit({"type": "stage", "key": pm.key, "status": "done", "report": final})

    report = {
        "meta": {
            "symbol": data.yahoo_symbol, "name": data.name, "market": data.market,
            "market_label": {"us": "미국", "kr": "한국", "cn": "중국 A주"}[data.market],
            "currency": data.currency, "price": data.last_close,
            "index_name": data.index_name, "coverage": _f(data.coverage * 100, 0),
            "issues": data.data_issues,
            "llm": _llm_meta(),
        },
        "agents": notes,
        "decision": final["decision"],
        "chart": _chart_payload(data),
    }
    emit({"type": "final", "report": report})
    return report


def _llm_meta() -> dict[str, Any]:
    llm = get_llm()
    return {"enabled": llm.enabled, "model": llm.model if llm.enabled else None}


def _chart_payload(data: StockData) -> dict[str, Any]:
    df = data.prices.tail(240)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    return {
        "dates": dates,
        "close": [round(float(v), 2) for v in df["close"]],
        "sma20": [None if pd.isna(v) else round(float(v), 2) for v in df["sma20"]],
        "sma60": [None if pd.isna(v) else round(float(v), 2) for v in df["sma60"]],
        "volume": [int(v) for v in df["volume"]],
    }
