import React, { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadialBarChart, RadialBar, PolarAngleAxis,
} from "recharts";

/* ---------- 유틸: 시드 기반 의사난수 (티커별 결정적 시뮬레이션) ---------- */
function hashCode(s: string) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}
function mulberry(seed: number) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- 시뮬레이션 데이터 생성 (yfinance 역할) ---------- */
function fetchStockData(tickerRaw: string) {
  const ticker = tickerRaw.trim().toUpperCase() || "AAPL";
  const rand = mulberry(hashCode(ticker));
  const base = 40 + rand() * 460;
  const drift = (rand() - 0.45) * 0.002;
  const vol = 0.008 + rand() * 0.03;

  const days = 90;
  const prices: { day: string; price: number; ma20: number | null }[] = [];
  const closes: number[] = [];
  let p = base;
  for (let i = 0; i < days; i++) {
    p = Math.max(1, p * (1 + drift + (rand() - 0.5) * vol * 2));
    closes.push(p);
    prices.push({
      day: `D-${days - i}`,
      price: Math.round(p * 100) / 100,
      ma20: null,
    });
  }
  for (let i = 19; i < days; i++) {
    const window = closes.slice(i - 19, i + 1);
    prices[i].ma20 = Math.round((window.reduce((a, b) => a + b, 0) / 20) * 100) / 100;
  }

  const price = closes[days - 1];
  const change = ((closes[days - 1] - closes[days - 2]) / closes[days - 2]) * 100;
  const high52 = Math.max(...closes) * (1 + rand() * 0.1);
  const low52 = Math.min(...closes) * (1 - rand() * 0.1);
  const sma20 = prices[days - 1].ma20 ?? price;
  const sma50 = closes.slice(-50).reduce((a, b) => a + b, 0) / 50;
  const rsi = (() => {
    let g = 0, l = 0;
    for (let i = days - 14; i < days; i++) {
      const d = closes[i] - closes[i - 1];
      if (d > 0) g += d; else l -= d;
    }
    const rs = l === 0 ? 100 : g / l;
    return Math.round(100 - 100 / (1 + rs));
  })();
  const macdBull = closes[days - 1] > sma20 && sma20 > sma50;

  return {
    ticker,
    price: Math.round(price * 100) / 100,
    change: Math.round(change * 100) / 100,
    high52: Math.round(high52),
    low52: Math.round(low52),
    sma20: Math.round(sma20 * 100) / 100,
    sma50: Math.round(sma50 * 100) / 100,
    rsi,
    macdBull,
    series: prices,
    volume: Math.round(1e6 + rand() * 5e7),
    peRatio: Math.round((8 + rand() * 60) * 10) / 10,
    epsGrowth: Math.round((rand() * 40 - 8) * 10) / 10,
    revenueGrowth: Math.round((rand() * 45 - 10) * 10) / 10,
    debtToEquity: Math.round(rand() * 180),
    profitMargin: Math.round((rand() * 42 - 4) * 10) / 10,
    roe: Math.round((rand() * 45 - 5) * 10) / 10,
    sector: ["Technology", "Semiconductors", "Consumer", "Healthcare", "Finance", "Energy"][
      Math.floor(rand() * 6)
    ],
    marketCap: Math.round(5 + rand() * 2900),
  };
}
type StockData = ReturnType<typeof fetchStockData>;

/* ---------- 에이전트 정의 ---------- */
type AgentId = "fetch" | "fundamental" | "technical" | "market" | "pm";
const AGENTS: { id: AgentId; name: string; role: string; color: string }[] = [
  { id: "fetch", name: "📊 Data Collector", role: "yfinance 시세·재무 데이터 수집", color: "#38bdf8" },
  { id: "fundamental", name: "🏛️ Fundamental Agent", role: "기본 분석 (재무제표, 밸류에이션)", color: "#a78bfa" },
  { id: "technical", name: "📈 Technical Agent", role: "기술 분석 (추세, 모멘텀, 지표)", color: "#34d399" },
  { id: "market", name: "🌐 Market Research Agent", role: "전반적 시장 환경 조사 (섹터·거시)", color: "#fbbf24" },
  { id: "pm", name: "🎯 Portfolio Manager", role: "종합 판단 및 최종 결정", color: "#f87171" },
];

function runFundamental(d: StockData) {
  let score = 50;
  const notes: string[] = [];
  if (d.epsGrowth > 15) { score += 15; notes.push(`EPS 성장률 ${d.epsGrowth}% — 강한 실적 모멘텀.`); }
  else if (d.epsGrowth < 0) { score -= 15; notes.push(`EPS 성장률 ${d.epsGrowth}% — 수익성 둔화 우려.`); }
  if (d.revenueGrowth > 20) { score += 10; notes.push(`매출 성장률 ${d.revenueGrowth}%로 확대 중.`); }
  else if (d.revenueGrowth < 0) { score -= 10; notes.push(`매출 성장률 ${d.revenueGrowth}% — 매출 축소.`); }
  if (d.peRatio < 20) { score += 8; notes.push(`P/E ${d.peRatio} — 합리적 밸류에이션.`); }
  else if (d.peRatio > 45) { score -= 8; notes.push(`P/E ${d.peRatio} — 밸류에이션 부담.`); }
  if (d.debtToEquity > 120) { score -= 8; notes.push(`부채비율(${d.debtToEquity}) 높은 레버리지.`); }
  if (d.profitMargin > 20) { score += 8; notes.push(`이익률 ${d.profitMargin}% — 우수한 수익 구조.`); }
  if (d.roe > 15) { score += 6; notes.push(`ROE ${d.roe}% — 자본 효율 양호.`); }
  return { score: Math.max(5, Math.min(98, Math.round(score))), verdict: score >= 60 ? "긍정적" : score >= 45 ? "중립적" : "부정적", notes };
}

function runTechnical(d: StockData) {
  let score = 50;
  const notes: string[] = [];
  if (d.price > d.sma20) { score += 10; notes.push(`현재가 $${d.price}가 20일 이동평균($${d.sma20}) 상회 — 단기 상승 추세.`); }
  else { score -= 10; notes.push(`현재가가 20일 이동평균($${d.sma20}) 하회 — 단기 하락 압력.`); }
  if (d.sma20 > d.sma50) { score += 8; notes.push("20일선 > 50일선 — 추세 정배열(골든크로스 구간)."); }
  else { score -= 8; notes.push("20일선 < 50일선 — 역배열(데드크로스 구간)."); }
  if (d.rsi > 70) { score -= 6; notes.push(`RSI ${d.rsi} — 과매수 구간, 조정 가능성.`); }
  else if (d.rsi < 30) { score += 6; notes.push(`RSI ${d.rsi} — 과매도 구간, 반등 잠재력.`); }
  else { notes.push(`RSI ${d.rsi} — 중립적 모멘텀.`); }
  if (d.macdBull) { score += 8; notes.push("MACD 시그널 상향 — 매수 신호."); }
  else { score -= 8; notes.push("MACD 시그널 하향 — 매도 신호."); }
  return { score: Math.max(5, Math.min(98, Math.round(score))), verdict: score >= 60 ? "강세" : score >= 45 ? "중립" : "약세", notes };
}

function runMarket(d: StockData) {
  const rand = mulberry(hashCode(d.ticker + "mkt"));
  const sectorMomentum = Math.round((rand() * 60 - 15) * 10) / 10;
  const ratePressure = rand() > 0.5;
  const sentiment = ["긍정적", "중립적", "신중한 낙관"][Math.floor(rand() * 3)];
  let score = 50 + Math.round(sectorMomentum / 3);
  if (ratePressure) { score -= 8; }
  if (d.marketCap > 800) { score += 5; }
  const notes = [
    `${d.sector} 섹터 모멘텀: ${sectorMomentum}% (30일 기준).`,
    ratePressure ? "금리 인상 압력으로 성장주 밸류에이션 부담." : "완화적 통화정책 기조가 위험자산에 유리.",
    `시장 심리: ${sentiment}. 기관 매수 비율 ${(40 + rand() * 40).toFixed(1)}%.`,
    `시가총액 $${d.marketCap}B — ${d.marketCap > 200 ? "대형주 안정성 확보" : "중소형 변동성 감안 필요"}.`,
  ];
  return { score: Math.max(5, Math.min(98, score)), verdict: score >= 60 ? "순풍" : score >= 45 ? "중립" : "역풍", notes };
}

function runPM(d: StockData, f: ReturnType<typeof runFundamental>, t: ReturnType<typeof runTechnical>, m: ReturnType<typeof runMarket>) {
  const composite = Math.round(f.score * 0.4 + t.score * 0.3 + m.score * 0.3);
  const action = composite >= 68 ? "강한 매수 (STRONG BUY)" : composite >= 55 ? "매수 (BUY)" : composite >= 42 ? "보유 (HOLD)" : composite >= 32 ? "매도 (SELL)" : "강한 매도 (STRONG SELL)";
  const upside = Math.round(((f.score * 0.5 + t.score * 0.3 + m.score * 0.2) / 100 - 0.5) * 40);
  const target = Math.round(d.price * (1 + upside / 100) * 100) / 100;
  const rationale = [
    `기본 분석 ${f.score}점(${f.verdict}) × 40% + 기술 분석 ${t.score}점(${t.verdict}) × 30% + 시장 조사 ${m.score}점(${m.verdict}) × 30% → 종합 ${composite}점.`,
    upside > 10 ? `3개 에이전트 의견이 대체로 긍정적이어서 상승 여력 ${upside}%를 반영한 목표가 $${target} 제시.` :
      upside > 0 ? `완만한 상승 여력(${upside}%) — 목표가 $${target}, 분할 매수 접근 권장.` :
      `하락 리스크가 우세(${upside}%) — 목표가 $${target}, 신중한 포지션 조정 필요.`,
    `리스크 관리: 손절 라인 $${(d.price * 0.92).toFixed(2)} (-8%), 포트폴리오 내 비중 5% 이내 권장.`,
  ];
  return { composite, action, target, rationale };
}

/* ---------- 진행 단계 로그 문구 ---------- */
const LOGS: Record<AgentId, string[]> = {
  fetch: ["yfinance 티커 조회…", "일봉 90일 / 재무 지표 다운로드…", "데이터 정규화 완료"],
  fundamental: ["손익계산서·밸류에이션 로드…", "EPS/매출 성장률 분석…", "기본 분석 리포트 작성"],
  technical: ["이동평균·RSI·MACD 계산…", "추세 구조 판별…", "기술 분석 리포트 작성"],
  market: ["섹터 모멘텀 스캔…", "거시·금리 환경 체크…", "시장 조사 리포트 작성"],
  pm: ["3개 에이전트 결과 취합…", "가중 종합 점수 산출…", "최종 결정 및 리스크 프레임 설정"],
};

/* ---------- UI 컴포넌트 ---------- */
function StatCard({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "up" | "down" }) {
  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-800/50 p-3">
      <div className="text-[11px] text-slate-400">{label}</div>
      <div className={`mt-0.5 text-lg font-semibold ${tone === "up" ? "text-emerald-400" : tone === "down" ? "text-rose-400" : "text-slate-100"}`}>{value}</div>
      {sub && <div className="text-[11px] text-slate-500">{sub}</div>}
    </div>
  );
}

function Gauge({ value, color, label }: { value: number; color: string; label: string }) {
  return (
    <div className="flex flex-col items-center">
      <div className="relative h-28 w-28">
        <ResponsiveContainer>
          <RadialBarChart innerRadius="72%" outerRadius="100%" data={[{ v: value }]} startAngle={90} endAngle={-270}>
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar dataKey="v" cornerRadius={12} fill={color} background={{ fill: "#1e293b" }} />
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex items-center justify-center text-xl font-bold text-slate-100">{value}</div>
      </div>
      <div className="mt-1 text-xs text-slate-400">{label}</div>
    </div>
  );
}

function AgentCard({ agent, status, log, active }: { agent: typeof AGENTS[0]; status: string; log: string[]; active: boolean }) {
  return (
    <div className={`rounded-xl border p-4 transition-all ${active ? "border-slate-500 bg-slate-800 shadow-lg shadow-slate-900/60" : "border-slate-700/60 bg-slate-800/30"} ${status === "done" ? "opacity-80" : ""}`}>
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold" style={{ color: agent.color }}>{agent.name}</span>
        {status === "done" && <span className="ml-auto text-xs text-emerald-400">✓ 완료</span>}
        {active && <span className="ml-auto text-xs text-sky-400 animate-pulse">● 실행 중</span>}
        {status === "pending" && <span className="ml-auto text-xs text-slate-500">대기</span>}
      </div>
      <div className="mt-1 text-[11px] text-slate-400">{agent.role}</div>
      {log.length > 0 && (
        <ul className="mt-2 space-y-1">
          {log.map((l, i) => (
            <li key={i} className="text-[11px] text-slate-300">› {l}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ReportSection({ title, color, verdict, score, notes }: { title: string; color: string; verdict: string; score: number; notes: string[] }) {
  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-800/40 p-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold" style={{ color }}>{title}</h4>
        <span className="rounded-full bg-slate-700 px-2 py-0.5 text-[11px] text-slate-200">{verdict} · {score}점</span>
      </div>
      <ul className="mt-2 space-y-1.5">
        {notes.map((n, i) => <li key={i} className="text-xs leading-relaxed text-slate-300">• {n}</li>)}
      </ul>
    </div>
  );
}

/* ---------- 메인 앱 ---------- */
export default function App() {
  const [input, setInput] = useState("AAPL");
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(-1);
  const [logs, setLogs] = useState<Record<AgentId, string[]>>({ fetch: [], fundamental: [], technical: [], market: [], pm: [] });
  const [data, setData] = useState<StockData | null>(null);
  const [results, setResults] = useState<{
    f: ReturnType<typeof runFundamental>; t: ReturnType<typeof runTechnical>;
    m: ReturnType<typeof runMarket>; pm: ReturnType<typeof runPM>;
  } | null>(null);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => () => { timers.current.forEach(clearTimeout); }, []);

  const start = () => {
    const t = input.trim().toUpperCase();
    if (!t) return;
    timers.current.forEach(clearTimeout); timers.current = [];
    setRunning(true); setStep(0);
    setLogs({ fetch: [], fundamental: [], technical: [], market: [], pm: [] });
    setResults(null);

    const d = fetchStockData(t);
    let elapsed = 0;
    AGENTS.forEach((agent, ai) => {
      LOGS[agent.id].forEach((line, li) => {
        elapsed += 380 + Math.random() * 250;
        timers.current.push(setTimeout(() => {
          setLogs((prev) => ({ ...prev, [agent.id]: [...prev[agent.id], line] }));
        }, elapsed));
      });
      elapsed += 500;
      timers.current.push(setTimeout(() => {
        if (agent.id === "fetch") setData(d);
        if (agent.id === "pm") {
          const f = runFundamental(d), tec = runTechnical(d), m = runMarket(d);
          setResults({ f, t: tec, m, pm: runPM(d, f, tec, m) });
        }
        if (ai < AGENTS.length - 1) setStep(ai + 1);
        else { setRunning(false); }
      }, elapsed));
    });
  };

  const stepStatus = (i: number): "pending" | "active" | "done" => {
    if (step === -1) return "pending";
    if (i < step) return "done";
    if (i === step) return running ? "active" : "done";
    return "pending";
  };

  return (
    <div className="min-h-screen bg-slate-900 p-6 text-slate-100">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6">
          <h1 className="text-2xl font-bold">🤖 AI 주식 분석 워크플로우</h1>
          <p className="mt-1 text-sm text-slate-400">
            데이터 수집 → 기본 분석 · 기술 분석 · 시장 조사 (병렬) → 포트폴리오 매니저 최종 결정 — 4단계 에이전트 체인
          </p>
        </header>

        {/* 파이프라인 다이어그램 */}
        <div className="mb-6 flex flex-wrap items-center gap-2 rounded-xl border border-slate-700/60 bg-slate-800/30 p-4">
          {AGENTS.map((a, i) => (
            <React.Fragment key={a.id}>
              <div
                className={`rounded-lg border px-3 py-2 text-xs font-medium transition-all ${
                  stepStatus(i) === "active" ? "border-sky-400 bg-sky-400/10 text-sky-300 animate-pulse"
                  : stepStatus(i) === "done" ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-300"
                  : "border-slate-700 bg-slate-800/50 text-slate-500"}`}
                style={stepStatus(i) !== "pending" ? { borderColor: a.color } : undefined}
              >
                {a.name.split(" ")[1] ?? a.name}
              </div>
              {i < AGENTS.length - 1 && <span className="text-slate-600">→</span>}
            </React.Fragment>
          ))}
        </div>

        {/* 입력 */}
        <div className="mb-6 flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !running && start()}
            placeholder="티커 입력 (예: AAPL, TSLA, NVDA, 005930.KS)"
            className="flex-1 rounded-lg border border-slate-700 bg-slate-800 px-4 py-2.5 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:border-sky-500"
          />
          <button
            onClick={start}
            disabled={running}
            className="rounded-lg bg-sky-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-sky-500 disabled:opacity-50"
          >
            {running ? "분석 중…" : "⚡ 워크플로우 실행"}
          </button>
        </div>

        {/* 에이전트 카드 */}
        <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {AGENTS.map((a, i) => (
            <AgentCard key={a.id} agent={a} status={stepStatus(i)} log={logs[a.id]} active={stepStatus(i) === "active"} />
          ))}
        </div>

        {/* 결과 */}
        {data && results && (
          <div className="space-y-5">
            {/* 요약 */}
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <StatCard label={`${data.ticker} 현재가`} value={`$${data.price}`} sub={`${data.change >= 0 ? "▲" : "▼"} ${Math.abs(data.change)}% · ${data.sector}`} tone={data.change >= 0 ? "up" : "down"} />
              <StatCard label="52주 범위" value={`$${data.low52} – $${data.high52}`} sub={`시총 $${data.marketCap}B`} />
              <StatCard label="목표가 (PM 제시)" value={`$${results.pm.target}`} sub="3개 에이전트 가중 반영" tone={results.pm.target >= data.price ? "up" : "down"} />
              <StatCard label="종합 점수" value={`${results.pm.composite} / 100`} sub={results.pm.action} />
            </div>

            {/* 결정 배너 */}
            <div className={`rounded-xl border p-4 ${results.pm.composite >= 55 ? "border-emerald-500/50 bg-emerald-500/10" : results.pm.composite >= 42 ? "border-amber-500/50 bg-amber-500/10" : "border-rose-500/50 bg-rose-500/10"}`}>
              <div className="text-xs text-slate-400">🎯 Portfolio Manager 최종 결정</div>
              <div className={`mt-1 text-2xl font-bold ${results.pm.composite >= 55 ? "text-emerald-400" : results.pm.composite >= 42 ? "text-amber-400" : "text-rose-400"}`}>{results.pm.action}</div>
              <ul className="mt-2 space-y-1">
                {results.pm.rationale.map((r, i) => <li key={i} className="text-xs text-slate-300">• {r}</li>)}
              </ul>
            </div>

            {/* 가격 차트 */}
            <div className="rounded-xl border border-slate-700/60 bg-slate-800/40 p-4">
              <h3 className="mb-3 text-sm font-semibold">90일 가격 추이 (데이터 수집 결과)</h3>
              <div className="h-56">
                <ResponsiveContainer>
                  <LineChart data={data.series}>
                    <XAxis dataKey="day" tick={{ fill: "#64748b", fontSize: 10 }} interval={14} axisLine={{ stroke: "#334155" }} />
                    <YAxis domain={["auto", "auto"]} tick={{ fill: "#64748b", fontSize: 10 }} axisLine={{ stroke: "#334155" }} width={60} />
                    <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }} />
                    <Line type="monotone" dataKey="price" stroke="#38bdf8" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="ma20" stroke="#f59e0b" strokeWidth={1.5} dot={false} strokeDasharray="4 3" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 flex gap-4 text-[11px] text-slate-400">
                <span>— <span className="text-sky-400">종가</span></span>
                <span>-- <span className="text-amber-400">20일 이동평균</span></span>
              </div>
            </div>

            {/* 게이지 + 지표 */}
            <div className="grid gap-3 lg:grid-cols-3">
              <div className="flex items-center justify-around rounded-xl border border-slate-700/60 bg-slate-800/40 p-4">
                <Gauge value={results.f.score} color="#a78bfa" label="기본 분석" />
                <Gauge value={results.t.score} color="#34d399" label="기술 분석" />
                <Gauge value={results.m.score} color="#fbbf24" label="시장 조사" />
              </div>
              <div className="lg:col-span-2 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <StatCard label="P/E" value={`${data.peRatio}x`} sub={`EPS 성장 ${data.epsGrowth}%`} />
                <StatCard label="RSI (14)" value={`${data.rsi}`} sub={data.rsi > 70 ? "과매수" : data.rsi < 30 ? "과매도" : "중립"} />
                <StatCard label="이익률 / ROE" value={`${data.profitMargin}% / ${data.roe}%`} sub={`부채비율 ${data.debtToEquity}`} />
                <StatCard label="매출 성장" value={`${data.revenueGrowth}%`} sub={`거래량 ${(data.volume / 1e6).toFixed(1)}M`} tone={data.revenueGrowth >= 0 ? "up" : "down"} />
              </div>
            </div>

            {/* 상세 리포트 */}
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-slate-200">📋 큐레이션 분석 보고서 — {data.ticker}</h3>
              <ReportSection title="🏛️ 기본 분석 (Fundamental Agent)" color="#a78bfa" verdict={results.f.verdict} score={results.f.score} notes={results.f.notes} />
              <ReportSection title="📈 기술 분석 (Technical Agent)" color="#34d399" verdict={results.t.verdict} score={results.t.score} notes={results.t.notes} />
              <ReportSection title="🌐 시장 조사 (Market Research Agent)" color="#fbbf24" verdict={results.m.verdict} score={results.m.score} notes={results.m.notes} />
              <ReportSection title="🎯 포트폴리오 매니저 종합 판단" color="#f87171" verdict={results.pm.action} score={results.pm.composite} notes={results.pm.rationale} />
            </div>

            <p className="pb-4 text-[11px] text-slate-500">
              ※ 데모용 시뮬레이션 데이터입니다. 실제 서비스 연결 시 Data Collector 에이전트를 yfinance API 호출로, 각 분석 에이전트를 Mistral API(agent 호출)로 교체하면 동일한 워크플로우로 실데이터 분석이 가능합니다. 투자 참고용이며 투자 권유가 아닙니다.
            </p>
          </div>
        )}

        {!data && !running && (
          <div className="rounded-xl border border-dashed border-slate-700 bg-slate-800/20 p-10 text-center text-sm text-slate-400">
            티커를 입력하고 <span className="font-semibold text-sky-400">워크플로우 실행</span>을 누르면 4단계 에이전트 체인이 순차 실행됩니다.
          </div>
        )}
      </div>
    </div>
  );
}