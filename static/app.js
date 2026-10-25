/* STOCK DESK — frontend logic: SSE pipeline + report rendering */

(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const STAGE_LABEL = {
    data: "데이터 수집",
    fundamental: "펀더멘털",
    technical: "테크니컬",
    market: "시장 리서치",
    portfolio: "포트폴리오",
  };

  let currentReport = null;
  let es = null;

  // ------------------------------------------------------------ init
  $("#edition-date").textContent = new Date().toLocaleDateString("ko-KR", {
    year: "numeric", month: "long", day: "numeric", weekday: "short",
  });

  $("#analyze-form").addEventListener("submit", (e) => {
    e.preventDefault();
    start($("#symbol-input").value.trim());
  });

  $$(".chip").forEach((btn) =>
    btn.addEventListener("click", () => {
      $("#symbol-input").value = btn.dataset.symbol;
      start(btn.dataset.symbol);
    })
  );

  $("#reset-btn").addEventListener("click", () => {
    if (es) es.close();
    $("#report-section").hidden = true;
    $("#howto").hidden = false;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  $("#download-btn").addEventListener("click", downloadMarkdown);

  // ------------------------------------------------------------ run
  function start(symbol) {
    if (!symbol) return;
    if (es) es.close();
    currentReport = null;

    $("#report-section").hidden = true;
    $("#howto").hidden = true;
    $("#notes").innerHTML = "";
    $("#chart-wrap").innerHTML = "";
    $("#risks-block").innerHTML = "";
    $("#log-body").innerHTML = "";
    $("#log-state").textContent = "running";
    $("#analyze-btn").disabled = true;

    $$(".stage").forEach((el) => {
      el.classList.remove("is-running", "is-done", "is-error");
      $(".stage-status .status-text", el).textContent = "대기";
      const sc = $(".stage-score", el);
      if (sc) sc.innerHTML = "";
    });

    log("system", `‘${symbol}’ 분석 요청 — 4-AGENT 파이프라인 가동`);
    window.scrollTo({ top: 0, behavior: "smooth" });

    es = new EventSource(`/api/analyze/stream?symbol=${encodeURIComponent(symbol)}`);

    es.addEventListener("stage", (e) => onStage(JSON.parse(e.data)));
    es.addEventListener("log", (e) => {
      const d = JSON.parse(e.data);
      log(d.stage || "system", d.message);
    });
    es.addEventListener("report", (e) => {
      const d = JSON.parse(e.data);
      renderStageScore(d.report);
    });
    es.addEventListener("final", (e) => {
      currentReport = JSON.parse(e.data).report;
      renderReport(currentReport);
      finish();
    });
    es.addEventListener("error", (e) => {
      let d = {};
      try { d = JSON.parse(e.data || "{}"); } catch (_) { /* 빈/불완전 data 무시 */ }
      // e.data 가 없으면 브라우저 네트워크 단절(프록시 타임아웃 등) — 서버 에러 이벤트가 아님
      const networkDrop = !e.data;
      log("error", d.message
        || (networkDrop
          ? "서버와의 연결이 끊겼습니다. 잠시 후 다시 시도해 주세요."
          : "알 수 없는 오류가 발생했습니다."));
      $("#log-state").textContent = "error";
      $("#analyze-btn").disabled = false;
      $$(".stage.is-running").forEach((el) => {
        el.classList.remove("is-running");
        el.classList.add("is-error");
      });
      if (es) es.close();
    });
    es.onerror = () => {
      if ($("#log-state").textContent === "running") {
        log("system", "연결이 종료되었습니다.");
        finish();
      }
    };
  }

  function finish() {
    $("#log-state").textContent = "done";
    $("#analyze-btn").disabled = false;
    if (es) { es.close(); es = null; }
  }

  function onStage(d) {
    const el = $(`.stage[data-stage="${d.key}"]`);
    if (!el) return;
    const txt = $(".stage-status .status-text", el);
    if (d.status === "running") {
      el.classList.add("is-running");
      el.classList.remove("is-done", "is-error");
      txt.textContent = "분석 중";
    } else if (d.status === "done") {
      el.classList.remove("is-running");
      el.classList.add("is-done");
      txt.textContent = "완료";
      if (d.meta) {
        log("data", `${d.meta.name} (${d.meta.symbol}) · ${d.meta.market.toUpperCase()} · ${d.meta.price.toLocaleString()} ${d.meta.currency}`);
      }
    }
  }

  function renderStageScore(report) {
    const el = $(`.stage[data-stage="${report.key}"]`);
    if (!el) return;
    const sc = $(".stage-score", el);
    if (sc && report.score != null) {
      sc.innerHTML = `${report.score.toFixed(0)}<span class="of">/100</span>`;
    }
  }

  // ------------------------------------------------------------ log
  function log(stage, message) {
    const body = $("#log-body");
    const ts = new Date().toLocaleTimeString("ko-KR", { hour12: false });
    const line = document.createElement("div");
    line.className = "log-line";
    line.innerHTML = `<span class="ts">${ts}</span><span class="tag">[${stage}]</span>${escapeHtml(message)}`;
    body.appendChild(line);
    while (body.children.length > 300) body.removeChild(body.firstChild);
    body.scrollTop = body.scrollHeight;
  }

  // ------------------------------------------------------------ report
  function renderReport(rep) {
    $("#report-section").hidden = false;
    const meta = rep.meta;
    const pm = rep.agents.find((a) => a.key === "portfolio");
    const d = rep.decision;
    const verdictClass = pm.score >= 60 ? "positive" : pm.score >= 44 ? "" : "neutral";

    $("#decision-cover").innerHTML = `
      <div class="cover-left">
        <div class="cover-kicker">FINAL DECISION · ${escapeHtml(meta.market_label)} · ${escapeHtml(meta.index_name)} CONTEXT</div>
        <div class="cover-name">${escapeHtml(meta.name)}</div>
        <div class="cover-ticker">${escapeHtml(meta.symbol)} · 데이터 커버리지 ${meta.coverage}% ·
          코멘티 ${meta.llm && meta.llm.enabled ? `LLM(${escapeHtml(meta.llm.model)})` : "규칙 기반"}</div>
        <div class="cover-price">${fmtPrice(meta.price, meta.currency)}
          <span class="metric-note">최근 종가</span>
        </div>
        <div class="stamp-row">
          <div class="stamp ${verdictClass}">${escapeHtml(d.verdict)}</div>
          <div>
            <div class="conviction">Conviction ${Math.round(pm.confidence * 100)}%</div>
            <div class="conviction-bar"><i style="width:${Math.round(pm.confidence * 100)}%"></i></div>
          </div>
        </div>
      </div>
      <div class="decision-table">
        ${dRow("액션", d.action)}
        ${dRow("포지션 비중", d.position, true)}
        ${dRow("진입 전략", d.entry)}
        ${dRow("손절 라인", d.stop, true)}
        ${dRow("목표가 1", d.target1, true)}
        ${dRow("목표가 2", d.target2, true)}
        ${dRow("보유 기간", d.horizon)}
        ${dRow("리스크 등급", `${d.risk_level} · ${d.daily_move}`)}
      </div>
      <div class="cover-summary">${escapeHtml(pm.summary)}</div>
    `;

    renderChart(rep.chart);

    $("#notes").innerHTML = rep.agents.map(noteHtml).join("");

    const allRisks = rep.agents.flatMap((a) => a.risks || []);
    $("#risks-block").innerHTML = `
      <h3>리스크 &amp; 유의사항</h3>
      <span class="section-note">각 에이전트가 식별한 리스크를 중복 제거해 요약합니다.</span>
      <ul class="point-list risk">${[...new Set(allRisks)].map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
    `;

    requestAnimationFrame(() =>
      $("#report-section").scrollIntoView({ behavior: "smooth", block: "start" })
    );
  }

  function dRow(label, value, num = false) {
    return `<div class="d-row"><div class="d-label">${label}</div>
      <div class="d-value ${num ? "num" : ""}">${escapeHtml(String(value ?? "—"))}</div></div>`;
  }

  function noteHtml(a) {
    const tone = a.score >= 60 ? "tone-good" : a.score >= 45 ? "tone-neutral" : "tone-bad";
    return `
      <article class="note">
        <div class="note-head">
          <span class="note-index">AGENT ${String(a.index).padStart(2, "0")}</span>
          <h3 class="note-title">${escapeHtml(a.title)}</h3>
          <span class="note-sub">${escapeHtml(a.subtitle)}</span>
          <span class="note-source ${a.source === "llm" ? "is-llm" : "is-rule"}">${a.source === "llm" ? "LLM 코멘티" : "규칙 기반 코멘티"}</span>
          <div class="note-stance ${tone}">
            <span class="score-big">${a.score.toFixed(1)}</span>${escapeHtml(a.stance)}
          </div>
        </div>
        <p class="note-summary">${escapeHtml(a.summary)}</p>
        <div class="metrics">
          ${(a.metrics || []).map(metricRow).join("") || '<div class="metric-row"><span class="metric-note">지표 없음</span></div>'}
        </div>
        <div class="note-points">
          <h4>판단 근거</h4>
          <ul class="point-list">${(a.bullets || []).map((b) => `<li>${escapeHtml(b)}</li>`).join("")}</ul>
          <div class="note-risks">
            <h4>이 에이전트의 경고</h4>
            <ul class="point-list risk">${(a.risks || []).map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
          </div>
        </div>
      </article>
    `;
  }

  function metricRow(m) {
    return `<div class="metric-row">
      <span class="metric-label">${escapeHtml(m.label)}</span>
      <span class="metric-value tone-${m.tone || "neutral"}">${escapeHtml(m.value)}</span>
      <span class="metric-note">${escapeHtml(m.note || "")}</span>
    </div>`;
  }

  // ------------------------------------------------------------ chart
  function renderChart(c) {
    if (!c || !c.close || !c.close.length) return;
    const W = 960, H = 320, padL = 8, padR = 62, padT = 18, volH = 62, padB = 26;
    const n = c.close.length;
    const prices = [...c.close, ...c.sma20.filter((v) => v != null), ...c.sma60.filter((v) => v != null)];
    const min = Math.min(...prices), max = Math.max(...prices);
    const span = max - min || 1;
    const plotH = H - padT - padB - volH - 12;
    const x = (i) => padL + (i / (n - 1)) * (W - padL - padR);
    const y = (v) => padT + plotH - ((v - min) / span) * plotH;
    const maxVol = Math.max(...c.volume, 1);
    const volY = (v) => padT + plotH + 12 + volH - (v / maxVol) * volH;

    const path = (arr) => {
      let d = "", pen = false;
      arr.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)} `;
        pen = true;
      });
      return d.trim();
    };

    const grid = [0, 0.25, 0.5, 0.75, 1].map((t) => {
      const v = min + span * t, yy = y(v);
      return `<line x1="${padL}" y1="${yy.toFixed(1)}" x2="${W - padR}" y2="${yy.toFixed(1)}"
                stroke="var(--rule-soft)" stroke-width="1"/>
              <text x="${W - padR + 8}" y="${(yy + 4).toFixed(1)}" font-family="var(--mono)"
                font-size="11" fill="var(--ink-faint)">${fmtShort(v)}</text>`;
    }).join("");

    const bars = c.volume.map((v, i) => {
      const w = Math.max(1, (W - padL - padR) / n * 0.62);
      return `<rect x="${(x(i) - w / 2).toFixed(1)}" y="${volY(v).toFixed(1)}" width="${w.toFixed(1)}"
                height="${(padT + plotH + 12 + volH - volY(v)).toFixed(1)}"
                fill="color-mix(in oklch, var(--ink) 16%, transparent)"/>`;
    }).join("");

    const dateLabels = [0, Math.floor(n / 2), n - 1].map((i) =>
      `<text x="${x(i).toFixed(1)}" y="${H - 6}" font-family="var(--mono)" font-size="11"
        fill="var(--ink-faint)" text-anchor="${i === 0 ? "start" : i === n - 1 ? "end" : "middle"}">${c.dates[i]}</text>`
    ).join("");

    $("#chart-wrap").innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="가격 차트">
        ${grid}
        ${bars}
        <path d="${path(c.sma60)}" fill="none" stroke="var(--vermillion)" stroke-width="1.6" stroke-dasharray="5 4"/>
        <path d="${path(c.sma20)}" fill="none" stroke="var(--ochre)" stroke-width="1.6"/>
        <path d="${path(c.close)}" fill="none" stroke="var(--ink)" stroke-width="2.2"/>
        ${dateLabels}
      </svg>
      <div class="chart-legend">
        <span><i style="background:var(--ink)"></i>종가</span>
        <span><i style="background:var(--ochre)"></i>SMA 20</span>
        <span><i style="background:var(--vermillion)"></i>SMA 60</span>
      </div>
    `;
  }

  // ------------------------------------------------------------ export
  function downloadMarkdown() {
    if (!currentReport) return;
    const rep = currentReport;
    const pm = rep.agents.find((a) => a.key === "portfolio");
    const d = rep.decision;
    const L = [];
    L.push(`# ${rep.meta.name} (${rep.meta.symbol}) — AI 리서치 노트`);
    L.push("");
    L.push(`- 시장: ${rep.meta.market_label} · 통화: ${rep.meta.currency}`);
    L.push(`- 기준일: ${new Date().toLocaleDateString("ko-KR")}`);
    L.push(`- 최근 종가: ${fmtPrice(rep.meta.price, rep.meta.currency)}`);
    L.push(`- 최종 판단: **${d.verdict}** (종합 ${pm.score.toFixed(1)}점 / 100, 신뢰도 ${Math.round(pm.confidence * 100)}%)`);
    L.push("");
    L.push("## 최종 결정");
    L.push(`| 항목 | 내용 |`);
    L.push(`| --- | --- |`);
    L.push(`| 액션 | ${d.action} |`);
    L.push(`| 포지션 비중 | ${d.position} |`);
    L.push(`| 진입 전략 | ${d.entry} |`);
    L.push(`| 손절 라인 | ${d.stop} |`);
    L.push(`| 목표가 1 / 2 | ${d.target1} / ${d.target2} |`);
    L.push(`| 보유 기간 | ${d.horizon} |`);
    L.push(`| 리스크 등급 | ${d.risk_level} (${d.daily_move}) |`);
    L.push("");
    L.push("## 요약");
    L.push(pm.summary);
    L.push("");
    for (const a of rep.agents) {
      L.push(`## AGENT ${String(a.index).padStart(2, "0")} — ${a.title} · ${a.stance} (${a.score.toFixed(1)}점)`);
      L.push(`
*코멘티 소스: ${a.source === "llm" ? `LLM(${rep.meta.llm?.model || "—"})` : "규칙 기반"}*`);
      L.push("");
      L.push(a.summary);
      L.push("");
      if (a.metrics?.length) {
        L.push(`| 지표 | 값 | 비고 |`);
        L.push(`| --- | --- | --- |`);
        for (const m of a.metrics) L.push(`| ${m.label} | ${m.value} | ${m.note || ""} |`);
        L.push("");
      }
      for (const b of a.bullets || []) L.push(`- ${b}`);
      for (const r of a.risks || []) L.push(`- ⚠️ ${r}`);
      L.push("");
    }
    L.push("> 본 문서는 공개 시장 데이터 기반 자동 생성 리서치이며 투자 권유가 아닙니다.");
    const blob = new Blob([L.join("\n")], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${rep.meta.symbol.replace(/\./g, "_")}_research_note.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ------------------------------------------------------------ utils
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  function fmtPrice(v, currency) {
    if (v == null) return "—";
    if (currency === "USD") return `$${Number(v).toLocaleString("en-US", { minimumFractionDigits: 2 })}`;
    const unit = currency === "KRW" ? "원" : currency === "CNY" ? "¥" : "";
    return `${Number(v).toLocaleString("ko-KR")}${unit}`;
  }

  function fmtShort(v) {
    if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return v.toFixed(v < 10 ? 2 : 0);
  }
})();
