"""Market data collection & normalization.

Supports:
  - US equities / ETFs      e.g. AAPL, NVDA, SPY
  - Korean equities         e.g. 005930.KS, 000660.KQ (6-digit bare code도 시도; Naver Finance로 보완)
  - China A-shares          e.g. 600519, 000001 (6-digit code; enriched via akshare)

Everything is normalized into one `StockData` object so downstream agents
never care where the numbers came from.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from . import indicators as ta

Emit = Callable[[dict], None]


class DataError(Exception):
    """Raised when a symbol cannot be resolved / fetched."""


MARKET_PRESETS = {
    "us": {"index": "^GSPC", "index_name": "S&P 500", "currency": "USD", "vol_index": "^VIX"},
    "kr": {"index": "^KS11", "index_name": "KOSPI", "currency": "KRW", "vol_index": None},
    "cn": {"index": "000001.SS", "index_name": "상해종합", "currency": "CNY", "vol_index": None},
}


# 중국 A주로 확정 라우팅되는 6자리 프리픽스: 상해 본부 60, 스타마켓 68, 상해 B주 90.
# 그 외 6자리(00/30/002/1~4 등)는 한국 종목(005930 등)일 수 있어 후보 라우팅에서 보완한다.
_CN_6DIGIT_PREFIXES = ("60", "68", "90")
_6DIGIT_RE = re.compile(r"\d{6}")


def kr_code6(symbol: str) -> Optional[str]:
    """'005930', '005930.KS' 등에서 6자리 숫자 종목 코드를 추출한다."""
    m = _6DIGIT_RE.search(symbol.strip().upper())
    return m.group(0) if m else None


def detect_market(symbol: str) -> str:
    s = symbol.strip().upper()
    if re.fullmatch(r"\d{6}", s):
        return "cn"
    if s.endswith(".KS") or s.endswith(".KQ"):
        return "kr"
    return "us"


def normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if re.fullmatch(r"\d{6}", s):
        # Yahoo convention: Shanghai = .SS, Shenzhen = .SZ
        return s + (".SS" if s[0] in "69" else ".SZ")
    return s


def route_candidates(raw: str) -> list[tuple[str, str]]:
    """한 종목 코드에 대해 시도할 (market, yahoo_symbol) 후보 목록.

    60/68/90으로 시작하는 6자리는 중국으로 확정한다. 그 외 6자리(예: 005930)는
    중국(.SZ/.SS) 시도 후 실패 시 한국(.KS → .KQ)으로도 시도해 사용자가 bare
    코드를 입력해도 한국 주식으로 정상 수집되도록 한다.
    """
    s = raw.strip().upper()
    primary = (detect_market(s), normalize_symbol(s))
    if primary[0] == "cn":
        code = kr_code6(s)
        if code and not code.startswith(_CN_6DIGIT_PREFIXES):
            return [primary, ("kr", code + ".KS"), ("kr", code + ".KQ")]
    return [primary]


@dataclass
class StockData:
    symbol: str            # as typed by the user
    yahoo_symbol: str      # resolved symbol actually fetched
    market: str
    name: str
    currency: str
    prices: pd.DataFrame   # date-indexed: open/high/low/close/volume + sma/ta columns
    profile: dict[str, Any]
    index_prices: pd.DataFrame
    index_name: str
    vol_index: Optional[pd.DataFrame]
    data_issues: list[str] = field(default_factory=list)

    @property
    def last_close(self) -> float:
        return float(self.prices["close"].iloc[-1])

    @property
    def coverage(self) -> float:
        keys = [
            "pe_ttm", "pb", "roe", "net_margin", "revenue_growth",
            "earnings_growth", "debt_to_equity", "dividend_yield", "market_cap",
        ]
        have = sum(1 for k in keys if self.profile.get(k) is not None)
        return have / len(keys)


def _blank_profile() -> dict[str, Any]:
    return {
        "name": None, "sector": None, "industry": None, "market_cap": None,
        "pe_ttm": None, "pe_forward": None, "pb": None, "roe": None,
        "gross_margin": None, "net_margin": None, "revenue_growth": None,
        "earnings_growth": None, "debt_to_equity": None, "dividend_yield": None,
        "beta": None, "week52_high": None, "week52_low": None,
    }


def _safe(v):
    try:
        if v is None:
            return None
        f = float(v)
        if not np.isfinite(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _kr_num(v) -> Optional[float]:
    """'9.58배' / '0.78%' / '215,500원' 같은 한글 단위 값을 float로 파싱한다."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s in ("-", "손실", "N/A"):
        return None
    s = s.replace("배", "").replace("%", "").replace("원", "").replace(",", "").strip()
    try:
        f = float(s)
    except ValueError:
        return None
    return f if np.isfinite(f) else None


def _kr_won(v) -> Optional[float]:
    """'1,248조 1,805억' / '3조 6,377억원' 같은 시총 표기를 원 단위 float로 파싱한다."""
    if v is None:
        return None
    s = str(v).replace(",", "").strip()
    total = 0.0
    found = False
    for unit, mult in (("조", 1e12), ("억", 1e8), ("만", 1e4)):
        m = re.search(r"([\d.]+)\s*" + unit, s)
        if m:
            total += float(m.group(1)) * mult
            found = True
    if not found:
        return _kr_num(v)
    return total if np.isfinite(total) else None


class DataCollector:
    """Fetches prices, fundamentals and market context for one symbol."""

    def collect(self, symbol: str, emit: Emit = lambda e: None) -> StockData:
        import yfinance as yf

        raw = symbol.strip().upper()
        if not raw:
            raise DataError("종목 코드를 입력해 주세요.")
        market = detect_market(raw)
        yahoo_symbol = normalize_symbol(raw)

        issues: list[str] = []

        # ---------- price history ----------
        candidates = route_candidates(raw)
        prices = None
        used = candidates[0]
        for idx, (cand_market, cand_symbol) in enumerate(candidates):
            emit({"type": "log", "stage": "data", "message": f"{cand_symbol} 일봉 2년치 수집 중…"})
            tries = 4 if idx == 0 else 2
            prices = self._fetch_history(yf, cand_symbol, tries=tries)
            if prices is None and cand_market == "cn":
                emit({"type": "log", "stage": "data", "message": "야후 응답 지연 → akshare로 전환합니다."})
                prices = self._fetch_akshare_hist(kr_code6(cand_symbol) or cand_symbol, tries=3 if idx == 0 else 1)
            if prices is not None:
                used = (cand_market, cand_symbol)
                break
        if prices is None:
            emit({"type": "log", "stage": "data", "message": "1차 수집 실패 → 소스 재시도(백오프) 후보를 다시 호출합니다."})
            cand_market, cand_symbol = candidates[0]
            prices = self._fetch_history(yf, cand_symbol)
            if prices is None and cand_market == "cn":
                prices = self._fetch_akshare_hist(kr_code6(cand_symbol) or cand_symbol)
        if prices is None or len(prices) < 40:
            raise DataError(
                f"'{symbol}' 가격 데이터를 가져오지 못했습니다. "
                "예: AAPL / 005930.KS / 600519 형태로 입력해 주세요."
            )
        market, yahoo_symbol = used
        preset = MARKET_PRESETS[market]
        prices = self._add_ta(prices)
        emit({
            "type": "log", "stage": "data",
            "message": f"가격 {len(prices)}일 확보 (최근 종가 {prices['close'].iloc[-1]:,.0f} {preset['currency']})",
        })

        # ---------- fundamentals ----------
        emit({"type": "log", "stage": "data", "message": "펀더멘털 지표 수집 중…"})
        profile = _blank_profile()
        profile.update(self._fetch_yf_profile(yf, yahoo_symbol, prices))
        if market == "cn":
            profile.update(self._fetch_akshare_profile(raw, profile))
        elif market == "kr":
            profile.update(self._fetch_naver_profile(kr_code6(raw) or raw, profile))
        self._fill_from_statements(yf, yahoo_symbol, profile, last_price=float(prices["close"].iloc[-1]))
        self._derive_from_prices(prices, profile)
        if profile.get("name") is None:
            profile["name"] = raw
        missing = [k for k in ("pe_ttm", "roe", "market_cap", "dividend_yield") if profile.get(k) is None]
        if missing:
            issues.append("일부 펀더멘털 지표가 소스에서 제공되지 않았습니다: " + ", ".join(missing))

        # ---------- market context ----------
        emit({"type": "log", "stage": "data", "message": f"{preset['index_name']} 지수/시장 컨텍스트 수집 중…"})
        index_prices = self._fetch_history(yf, preset["index"], period="2y")
        if index_prices is None:
            index_prices = self._fetch_history(yf, preset["index"], period="1y")
        if index_prices is None:
            issues.append("시장 지수 데이터를 가져오지 못해 시장 리서치가 제한됩니다.")
            index_prices = pd.DataFrame(
                {"open": [], "high": [], "low": [], "close": [], "volume": []}
            )
        vol_index = None
        if preset["vol_index"]:
            vol_index = self._fetch_history(yf, preset["vol_index"], period="1y")

        return StockData(
            symbol=raw,
            yahoo_symbol=yahoo_symbol,
            market=market,
            name=str(profile.get("name") or raw),
            currency=preset["currency"],
            prices=prices,
            profile=profile,
            index_prices=index_prices,
            index_name=preset["index_name"],
            vol_index=vol_index,
            data_issues=issues,
        )

    # ------------------------------------------------------------------
    def _fetch_history(self, yf, symbol: str, period: str = "2y", tries: int = 4) -> Optional[pd.DataFrame]:
        import time

        for attempt in range(tries):
            try:
                df = yf.Ticker(symbol).history(period=period, auto_adjust=True, actions=False)
                if df is not None and not df.empty:
                    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df = df.dropna(subset=["close"]).sort_index()
                    return df
            except Exception:
                pass
            if attempt < tries - 1:
                time.sleep(1.5 * (attempt + 1))
        return None

    def _fetch_akshare_hist(self, code: str, tries: int = 3) -> Optional[pd.DataFrame]:
        import time

        for attempt in range(tries):
            try:
                import akshare as ak

                end = dt.date.today()
                start = end - dt.timedelta(days=900)
                df = ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
                    adjust="qfq",
                )
                if df is not None and not df.empty:
                    out = pd.DataFrame({
                        "open": pd.to_numeric(df["开盘"], errors="coerce"),
                        "high": pd.to_numeric(df["最高"], errors="coerce"),
                        "low": pd.to_numeric(df["最低"], errors="coerce"),
                        "close": pd.to_numeric(df["收盘"], errors="coerce"),
                        "volume": pd.to_numeric(df["成交量"], errors="coerce"),
                    }, index=pd.to_datetime(df["日期"]))
                    return out.dropna(subset=["close"]).sort_index()
            except Exception:
                pass
            if attempt < tries - 1:
                time.sleep(3.0 * (attempt + 1))
        return None

    def _add_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out["close"]
        out["sma20"] = ta.sma(close, 20)
        out["sma60"] = ta.sma(close, 60)
        out["sma120"] = ta.sma(close, 120)
        out["ema20"] = ta.ema(close, 20)
        out["rsi14"] = ta.rsi(close, 14)
        macd_line, signal_line, hist = ta.macd(close)
        out["macd"] = macd_line
        out["macd_signal"] = signal_line
        out["macd_hist"] = hist
        up, mid, low, width = ta.bollinger(close)
        out["bb_upper"] = up
        out["bb_mid"] = mid
        out["bb_lower"] = low
        out["bb_width"] = width
        out["atr14"] = ta.atr(out, 14)
        return out

    # ------------------------------------------------------------------
    def _fetch_yf_profile(self, yf, symbol: str, prices: pd.DataFrame) -> dict[str, Any]:
        info: dict[str, Any] = {}
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception:
            info = {}

        def pct(key):
            v = _safe(info.get(key))
            return None if v is None else v

        price = _safe(prices["close"].iloc[-1]) if len(prices) else None

        pe_ttm = _safe(info.get("trailingPE"))
        pe_forward = _safe(info.get("forwardPE"))
        pb = _safe(info.get("priceToBook"))
        eps_ttm = _safe(info.get("trailingEps"))
        eps_forward = _safe(info.get("forwardEps"))
        bps = _safe(info.get("bookValue"))

        # Yahoo가 trailingPE 등을 비워두는 경우(한국 주식에서 흔함) EPS/BPS로 역산한다.
        if pe_ttm is None and price is not None and eps_ttm is not None and eps_ttm > 0:
            pe_ttm = price / eps_ttm
        if pe_forward is None and price is not None and eps_forward is not None and eps_forward > 0:
            pe_forward = price / eps_forward
        if pb is None and price is not None and bps is not None and bps > 0:
            pb = price / bps

        gross = _safe(info.get("grossMargins"))
        return {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": _safe(info.get("marketCap")),
            "pe_ttm": pe_ttm,
            "pe_forward": pe_forward,
            "pb": pb,
            "roe": pct("returnOnEquity"),
            "gross_margin": gross,
            "net_margin": pct("profitMargins"),
            "revenue_growth": pct("revenueGrowth"),
            "earnings_growth": pct("earningsGrowth"),
            "debt_to_equity": pct("debtToEquity"),
            "dividend_yield": pct("dividendYield"),
            "beta": _safe(info.get("beta")),
            "week52_high": _safe(info.get("fiftyTwoWeekHigh")),
            "week52_low": _safe(info.get("fiftyTwoWeekLow")),
        }

    def _fetch_akshare_profile(self, code: str, profile: dict[str, Any]) -> dict[str, Any]:
        try:
            import akshare as ak

            df = ak.stock_individual_info_em(symbol=code)
            if df is None or df.empty:
                return {}
            mapping = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1]))
            out: dict[str, Any] = {}
            for k, v in mapping.items():
                kl = str(k)
                if "股票简称" in kl or "名称" in kl:
                    out["name"] = str(v)
                elif "行业" in kl:
                    out["industry"] = str(v)
                elif "市盈率" in kl and profile.get("pe_ttm") is None:
                    out["pe_ttm"] = _safe(v)
                elif "市净率" in kl and profile.get("pb") is None:
                    out["pb"] = _safe(v)
                elif "总市值" in kl and profile.get("market_cap") is None:
                    out["market_cap"] = _safe(v)
            return out
        except Exception:
            return {}

    def _fetch_naver_profile(self, code: str, profile: dict[str, Any]) -> dict[str, Any]:
        """네이버 금융 모바일 API로 한국 주식 PER/PBR/배당수익률/시총을 보완한다.

        yfinance는 005930.KS 등 한국 종목의 trailingPE를 비워두는 경우가 많아
        밸류에이션 평가가 누락된다. 여기서 기존 값이 비어 있을 때만 채운다.
        """
        import requests

        m = _6DIGIT_RE.search(code.strip().upper())
        if not m:
            return {}
        out: dict[str, Any] = {}
        try:
            resp = requests.get(
                f"https://m.stock.naver.com/api/stock/{m.group(0)}/integration",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=4,
            )
            if resp.status_code != 200:
                return {}
            data = resp.json()
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}

        name = data.get("stockName")
        if name:
            out["name"] = str(name)

        values: dict[str, Any] = {}
        for item in data.get("totalInfos") or []:
            if isinstance(item, dict) and item.get("key") is not None:
                values[str(item["key"])] = item.get("value")

        if profile.get("pe_ttm") is None:
            v = _kr_num(values.get("PER"))
            if v is not None:
                out["pe_ttm"] = v
        if profile.get("pe_forward") is None:
            v = _kr_num(values.get("추정PER"))
            if v is not None:
                out["pe_forward"] = v
        if profile.get("pb") is None:
            v = _kr_num(values.get("PBR"))
            if v is not None:
                out["pb"] = v
        if profile.get("dividend_yield") is None:
            v = _kr_num(values.get("배당수익률"))
            if v is not None:
                out["dividend_yield"] = v / 100  # 0.78% → 0.0078 (비율)
        if profile.get("market_cap") is None:
            v = _kr_won(values.get("시총"))
            if v is not None:
                out["market_cap"] = v
        return out

    def _fill_from_statements(self, yf, symbol: str, profile: dict[str, Any], last_price: Optional[float] = None) -> None:
        """Derive growth / profitability / leverage when `info` is thin (common for KR)."""
        try:
            t = yf.Ticker(symbol)
            inc = t.income_stmt
            bs = t.balance_sheet
        except Exception:
            return
        try:
            # yfinance 재무제표 컬럼은 최신 연도가 앞(iloc[0])에 오는 역순이다.
            # 날짜 오름차순(과거 → 최신)으로 정렬해 iloc[-1]이 항상 최신 실적이도록 보장.
            if inc is not None and not inc.empty:
                inc = inc.sort_index(axis=1)
            if bs is not None and not bs.empty:
                bs = bs.sort_index(axis=1)
            if inc is not None and not inc.empty:
                if "Total Revenue" in inc.index:
                    rev = inc.loc["Total Revenue"].dropna()
                    if len(rev) >= 2 and rev.iloc[-2] not in (0, None):
                        if profile.get("revenue_growth") is None:
                            profile["revenue_growth"] = _safe(rev.iloc[-1] / rev.iloc[-2] - 1)
                if "Net Income" in inc.index and "Total Revenue" in inc.index:
                    ni = inc.loc["Net Income"].dropna()
                    rev = inc.loc["Total Revenue"].dropna()
                    if len(ni) and len(rev) and rev.iloc[-1]:
                        if profile.get("net_margin") is None:
                            profile["net_margin"] = _safe(ni.iloc[-1] / rev.iloc[-1])
                        if profile.get("earnings_growth") is None and len(ni) >= 2 and ni.iloc[-2]:
                            profile["earnings_growth"] = _safe(ni.iloc[-1] / ni.iloc[-2] - 1)
                # PER 역산: ① 주가 / EPS, ② 시총 / 당기순이익
                if profile.get("pe_ttm") is None:
                    for row in ("Diluted EPS", "Basic EPS"):
                        if row not in inc.index:
                            continue
                        eps = inc.loc[row].dropna()
                        if len(eps) and last_price is not None and _safe(eps.iloc[-1]) and _safe(eps.iloc[-1]) > 0:
                            profile["pe_ttm"] = _safe(last_price / _safe(eps.iloc[-1]))
                            break
                    if profile.get("pe_ttm") is None and "Net Income" in inc.index:
                        mc = _safe(profile.get("market_cap"))
                        ni = inc.loc["Net Income"].dropna()
                        if mc is not None and mc > 0 and len(ni) and _safe(ni.iloc[-1]) and ni.iloc[-1] > 0:
                            profile["pe_ttm"] = _safe(mc / ni.iloc[-1])
                if "Gross Profit" in inc.index and "Total Revenue" in inc.index:
                    gp = inc.loc["Gross Profit"].dropna()
                    rev = inc.loc["Total Revenue"].dropna()
                    if len(gp) and len(rev) and rev.iloc[-1]:
                        if profile.get("gross_margin") is None:
                            profile["gross_margin"] = _safe(gp.iloc[-1] / rev.iloc[-1])
            if bs is not None and not bs.empty:
                if "Common Stock Equity" in bs.index:
                    eq = bs.loc["Common Stock Equity"].dropna()
                    if inc is not None and "Net Income" in inc.index and len(eq):
                        ni = inc.loc["Net Income"].dropna()
                        if len(ni) and eq.iloc[-1]:
                            if profile.get("roe") is None:
                                profile["roe"] = _safe(ni.iloc[-1] / eq.iloc[-1])
                    # PBR 역산: 시총 / 자기자본
                    if profile.get("pb") is None and len(eq):
                        mc = _safe(profile.get("market_cap"))
                        if mc is not None and mc > 0 and _safe(eq.iloc[-1]) and eq.iloc[-1] > 0:
                            profile["pb"] = _safe(mc / eq.iloc[-1])
                    if "Total Debt" in bs.index and len(eq) and eq.iloc[-1]:
                        debt = bs.loc["Total Debt"].dropna()
                        if len(debt) and profile.get("debt_to_equity") is None:
                            profile["debt_to_equity"] = _safe(debt.iloc[-1] / eq.iloc[-1] * 100)
        except Exception:
            return

    def _derive_from_prices(self, prices: pd.DataFrame, profile: dict[str, Any]) -> None:
        if profile.get("week52_high") is None:
            profile["week52_high"] = float(prices["high"].tail(252).max())
        if profile.get("week52_low") is None:
            profile["week52_low"] = float(prices["low"].tail(252).min())
