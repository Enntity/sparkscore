// SPDX-License-Identifier: AGPL-3.0-only
// SparkScore live demo. Every probability and every millisecond on screen
// comes from a live /v1/systemone call; nothing is precomputed or overridden.
"use strict";

const P = new URLSearchParams(location.search);
const AUTO = P.get("autoplay") === "1";
const SCENE = P.get("scene") || "home";
const stage = document.getElementById("stage");
const root = document.getElementById("scene");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const f2 = (p) => p.toFixed(2);
const ms0 = (v) => `${Math.round(v)}`;

const flyLayer = document.getElementById("fly");
function stageRect(e) {
  const s = stage.getBoundingClientRect(), r = e.getBoundingClientRect();
  return { x: (r.left - s.left) / SCALE, y: (r.top - s.top) / SCALE, w: r.width / SCALE, h: r.height / SCALE };
}
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}
let SCALE = 1;
function fit() {
  SCALE = Math.min(innerWidth / 1920, innerHeight / 1080);
  stage.style.setProperty("--s", SCALE);
}
addEventListener("resize", fit);
fit();

// ------------------------------------------------------------- the endpoint
window.__calls = [];
async function systemOne(scene, state, questions, extra = {}) {
  for (let attempt = 0; attempt < 30; attempt++) {
    const t0 = performance.now();
    const r = await fetch("/v1/systemone", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "sparkscore-d3", state, questions, ...extra }),
    });
    if (r.status === 529) { // every slot is busy: back off as the server asks
      await sleep(1000 * (Number(r.headers.get("retry-after")) || 1));
      continue;
    }
    const body = await r.json();
    const rt = performance.now() - t0;
    if (!r.ok) throw new Error(`systemone ${r.status}: ${JSON.stringify(body.detail)}`);
    const timing = {};
    let pack = null;
    for (const part of (r.headers.get("server-timing") || "").split(",")) {
      const m = /\s*(\w+);dur=([\d.]+)/.exec(part);
      if (m) timing[m[1]] = Number(m[2]);
      const k = /\s*pack;desc=(\w+)/.exec(part);
      if (k) pack = k[1];
    }
    const out = {
      answers: body.answers, server: timing.total, backend: timing.backend, rt,
      passes: Number(r.headers.get("x-sparkscore-passes")), tokens: body.usage.input_tokens,
      questions: Object.keys(questions).length,
      pack, packTokens: Number(r.headers.get("x-sparkscore-pack-tokens")) || 0,
    };
    window.__calls.push({ scene, server: out.server, rt: out.rt, passes: out.passes, questions: out.questions, tokens: out.tokens, t: Date.now() });
    return out;
  }
  throw new Error("systemone: every slot stayed busy");
}

class Stats {
  constructor() { this.server = []; this.rt = []; this.events = []; this.decisions = 0; }
  add(r, rated = true) {
    this.server.push(r.server); this.rt.push(r.rt);
    this.decisions += r.questions;
    if (rated) this.events.push([performance.now(), r.questions]);
  }
  // Slow-motion playback stops the rate clock, so the rate shows the model, not the replay.
  pause() { this.pausedAt = performance.now(); }
  resume() { const d = performance.now() - this.pausedAt; for (const e of this.events) e[0] += d; }
  static pct(arr, q) {
    if (!arr.length) return NaN;
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.max(0, Math.ceil(q * s.length) - 1))];
  }
  p(q) { return Stats.pct(this.server, q); }
  prt(q) { return Stats.pct(this.rt, q); }
  rate(windowMs = 4000) { // decisions per second over the last window
    const now = performance.now();
    const recent = this.events.filter(([t]) => now - t <= windowMs);
    if (recent.length < 2) return 0;
    const span = Math.max(1000, now - recent[0][0]);
    return recent.reduce((s, [, n]) => s + n, 0) / (span / 1000);
  }
}

// ------------------------------------------------------------------ chrome
const capEl = document.getElementById("caption");
let capTimer = null;
async function caption(main, sub = "", hold = 3400) {
  clearTimeout(capTimer);
  if (capEl.classList.contains("on")) { capEl.classList.remove("on"); await sleep(380); }
  capEl.querySelector(".cap-main").innerHTML = main;
  capEl.querySelector(".cap-sub").textContent = sub;
  capEl.classList.add("on");
  if (hold) capTimer = setTimeout(() => capEl.classList.remove("on"), hold);
}
function title(t) { document.getElementById("sceneTitle").textContent = t; }
function done() { window.__demoDone = true; }

async function health() {
  const chip = document.getElementById("live");
  try {
    const r = await fetch("/health");
    chip.classList.toggle("down", !r.ok);
    if (!r.ok) chip.lastChild.textContent = "OFFLINE";
  } catch {
    chip.classList.add("down");
    chip.lastChild.textContent = "OFFLINE";
  }
}

function hud(tiles) {
  const h = el("div", "hud");
  const out = {};
  for (const [key, label, color, sub] of tiles) {
    const t = el("div", "tile", `<div class="k">${label}</div><div class="v">—</div><div class="sub">${sub || ""}</div>`);
    t.style.setProperty("--c", color);
    h.append(t);
    out[key] = (v, unit = "") => { t.querySelector(".v").innerHTML = `${v}${unit ? `<small>${unit}</small>` : ""}`; };
    out[key].sub = (s) => { t.querySelector(".sub").textContent = s; };
  }
  root.append(h);
  return out;
}

// ----------------------------------------------------------------- widgets
function pbar(label, color) {
  const e = el("div", "pbar", `<div class="l">${esc(label)}</div><div class="t"><div class="f"></div></div><div class="n">—</div>`);
  if (color) e.style.setProperty("--c", color);
  e.set = (p, win) => {
    e.style.setProperty("--v", p);
    e.querySelector(".n").textContent = f2(p);
    e.classList.toggle("win", win === true);
    e.classList.toggle("lose", win === false);
  };
  return e;
}
function levels(labels, colors) {
  const e = el("div", "levels");
  labels.forEach((l, i) => {
    const d = el("div", "lv", `<span>${esc(l)}</span>`);
    d.style.setProperty("--c", colors[i]);
    e.append(d);
  });
  e.set = (probs) => {
    const arr = labels.map((_, i) => probs[String(i)] || 0);
    const best = argmax(arr);
    [...e.children].forEach((d, i) => { d.style.setProperty("--p", arr[i]); d.classList.toggle("on", i === best && arr[i] > 0); });
    return best;
  };
  return e;
}
function ring(cls, r, stroke, label) {
  const C = 2 * Math.PI * r, size = 2 * (r + stroke);
  const e = el("div", cls, `<svg viewBox="0 0 ${size} ${size}"><circle class="trk" cx="${size / 2}" cy="${size / 2}" r="${r}" stroke-width="${stroke}"/><circle class="arc" cx="${size / 2}" cy="${size / 2}" r="${r}" stroke-width="${stroke}" stroke-dasharray="${C}" stroke-dashoffset="${C}"/></svg><div class="rv"><b>—</b><span>${label}</span></div>`);
  const arc = e.querySelector(".arc");
  e.set = (p, color) => {
    arc.style.strokeDashoffset = C * (1 - p);
    e.style.setProperty("--c", color);
    e.querySelector("b").textContent = f2(p);
  };
  return e;
}
const URG_COLORS = ["#5d6890", "var(--cyan)", "var(--amber)", "var(--red)"];

// ===================================================== 1. Inbox Firehose
async function sceneInbox() {
  title("Inbox Firehose");
  root.className = "inbox";
  const D = DEMO.inbox;
  const LC = { billing: "var(--amber)", infra: "var(--cyan)", access: "var(--violet)", sales: "var(--green)", product: "var(--pink)" };
  const ULAB = ["Can wait", "This week", "Today", "Right now"];
  const H = hud([
    ["rate", "Decisions / sec", "var(--cyan)", "last 4 s"],
    ["total", "Decisions", "var(--violet)", ""],
    ["p50", "Server p50", "var(--green)", ""],
    ["p95", "Server p95", "var(--amber)", ""],
    ["passes", "Passes / ticket", "var(--pink)", "measured"],
  ]);

  const insp = el("div", "panel insp");
  insp.innerHTML = `<div class="insp-head"><span class="eyebrow" id="ie">Now scoring</span><span class="ms" id="ims">— ms</span></div>
    <div class="who" id="iwho"></div><div class="ttext" id="itext"></div>`;
  const qTeam = el("div", "qblock", `<div class="ql">Which team? <span class="qtype">choice</span></div>`);
  const teamBars = {};
  for (const [k, name] of Object.entries(D.lanes)) { teamBars[k] = pbar(name, LC[k]); teamBars[k].style.setProperty("--lw", "190px"); qTeam.append(teamBars[k]); }
  const qUrg = el("div", "qblock", `<div class="ql">Urgency <span class="qtype">score</span></div>`);
  const urg = levels(ULAB, URG_COLORS); qUrg.append(urg);
  const qFlags = el("div", "qblock", `<div class="ql">Flags <span class="qtype">noul</span></div>`);
  const refundBar = pbar("Refund allowed", "var(--green)"); refundBar.style.setProperty("--lw", "190px");
  const humanBar = pbar("Needs a human", "var(--pink)"); humanBar.style.setProperty("--lw", "190px");
  qFlags.append(refundBar, humanBar);
  insp.append(qTeam, qUrg, qFlags);
  root.append(insp);

  const lanesEl = el("div", "lanes");
  const lanes = {};
  for (const [k, name] of Object.entries(D.lanes)) {
    const lane = el("div", "lane", `<div class="lane-h"><b>${name}</b><span>0</span></div><div class="lane-body"></div>`);
    lane.style.setProperty("--c", LC[k]);
    lanes[k] = { el: lane, body: lane.querySelector(".lane-body"), count: 0 };
    lanesEl.append(lane);
  }
  root.append(lanesEl);

  const stats = new Stats();
  const $ = (id) => insp.querySelector(id);
  const short = (n) => { const [a, b] = n.split(" "); return b ? `${a} ${b[0]}.` : a; };

  function showTicket(t, label) {
    $("#ie").textContent = label;
    $("#iwho").innerHTML = `${esc(t.name)}<span class="plan">${esc(t.plan)}</span>`;
    $("#itext").textContent = `“${t.text}”`;
  }
  function showAnswers(r) {
    const a = r.answers;
    const team = a.team.choice;
    for (const [k, b] of Object.entries(teamBars)) b.set(a.team.probabilities[k], k === team);
    urg.set(a.urgency.probabilities);
    refundBar.set(a.refund.noul, a.refund.noul >= 0.5);
    humanBar.set(a.human.noul, a.human.noul >= 0.5);
    $("#ims").textContent = `${ms0(r.server)} ms`;
  }
  function flyToLane(t, r, quick) {
    const a = r.answers, lane = lanes[a.team.choice];
    const level = argmax([0, 1, 2, 3].map((i) => a.urgency.probabilities[String(i)]));
    const card = el("div", "lcard", `
      <div class="lc-top"><b>${esc(short(t.name))}</b><span class="ms">${ms0(r.server)} ms</span></div>
      <div class="lc-urg"><span class="pill">${ULAB[level]}</span></div>
      <div class="lc-row"><span>refund</span><div class="t"><i></i></div><em>${f2(a.refund.noul)}</em></div>
      <div class="lc-row"><span>human</span><div class="t"><i></i></div><em>${f2(a.human.noul)}</em></div>`);
    card.style.setProperty("--c", LC[a.team.choice]);
    card.style.setProperty("--uc", URG_COLORS[level]);
    const rows = card.querySelectorAll(".lc-row");
    rows[0].style.setProperty("--v", a.refund.noul); rows[0].style.setProperty("--bc", "var(--green)");
    rows[1].style.setProperty("--v", a.human.noul); rows[1].style.setProperty("--bc", "var(--pink)");
    // FLIP: remember where the lane's cards were, insert, then animate.
    const before = new Map([...lane.body.children].map((c) => [c, c.getBoundingClientRect().top]));
    lane.body.prepend(card);
    for (const [c, top] of before) {
      const dy = (top - c.getBoundingClientRect().top) / SCALE;
      if (dy) c.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: quick ? 320 : 520, easing: "cubic-bezier(.2,.8,.2,1)" });
    }
    // The card flies from the inspector in an overlay above the lanes, which clip.
    const from = stageRect(insp), to = stageRect(card);
    const ghost = card.cloneNode(true);
    Object.assign(ghost.style, { position: "absolute", left: `${to.x}px`, top: `${to.y}px`, width: `${to.w}px`, margin: "0" });
    flyLayer.append(ghost);
    card.style.visibility = "hidden";
    // Launch from just right of the inspector, growing rightwards, so the
    // flight never crosses the inspector's text.
    ghost.style.transformOrigin = "left center";
    const dx = from.x + from.w + 14 - to.x, dy = from.y + 110 - (to.y + to.h / 2);
    ghost.animate([
      { transform: `translate(${dx}px, ${dy}px) scale(1.25)`, opacity: 0, filter: "blur(6px)" },
      { offset: 0.22, opacity: 1, filter: "blur(0)" },
      { transform: "none", opacity: 1 },
    ], { duration: quick ? 520 : 860, easing: "cubic-bezier(.3,.7,.2,1)" }).onfinish = () => {
      card.style.visibility = "";
      ghost.remove();
    };
    lane.count++;
    lane.el.querySelector(".lane-h span").textContent = lane.count;
    lane.el.classList.remove("hit"); void lane.el.offsetWidth; lane.el.classList.add("hit");
    while (lane.body.children.length > 7) lane.body.lastChild.remove();
  }
  function updateHud(r, slow = false) {
    stats.add(r, !slow);
    const rate = stats.rate();
    if (!slow) H.rate(rate ? rate.toFixed(1) : "—");
    H.total(stats.decisions.toLocaleString("en-US"));
    H.total.sub(`${stats.server.length} tickets`);
    H.p50(ms0(stats.p(0.5)), "ms");
    H.p50.sub(`round trip ${ms0(stats.prt(0.5))} ms`);
    H.p95(ms0(stats.p(0.95)), "ms");
    H.passes(r.passes, `× ${r.questions} questions`);
  }

  let i = 0;
  const next = () => D.tickets[i++ % D.tickets.length];
  const t0 = performance.now();
  const elapsed = () => performance.now() - t0;

  // Full speed: one request in flight, back to back, live.
  async function fullSpeed(forMs, caps) {
    const start = elapsed();
    while (elapsed() - start < forMs) {
      while (caps.length && elapsed() - start >= caps[0][0]) caps.shift()[1]();
      const t = next();
      const r = await systemOne("inbox", D.state(t), D.questions);
      showTicket(t, "Last decision");
      showAnswers(r);
      updateHud(r);
      flyToLane(t, r, true);
      await sleep(16);
    }
  }

  // Start at full speed, so nothing reads as a warm-up.
  caption("<b>One call per ticket.</b> Four typed questions.", "team (choice) · urgency (score) · refund under the policy (noul) · needs a human (noul)", 5600);
  await fullSpeed(6500, []);

  // Slow the playback, not the model, so a viewer can read one decision.
  insp.classList.add("slowmo");
  stats.pause();
  caption("<b>Slow motion</b>, so you can read one", "the playback is slowed down; the model is not", 0);
  for (let k = 0; k < 2; k++) {
    const t = next();
    showTicket(t, "Now scoring");
    insp.querySelector("#ims").textContent = "— ms";
    insp.classList.add("scan");
    await sleep(350);
    const r = await systemOne("inbox", D.state(t), D.questions);
    insp.classList.remove("scan");
    insp.classList.remove("flash"); void insp.offsetWidth; insp.classList.add("flash");
    showAnswers(r);
    insp.querySelector("#ie").textContent = `Decided in ${ms0(r.server)} ms`;
    updateHud(r, true);
    await sleep(k === 0 ? 2200 : 1600);
    flyToLane(t, r, false);
    await sleep(250);
  }
  insp.classList.remove("slowmo");
  stats.resume();

  caption("Back to <b>full speed</b>", "one request in flight, back to back, live", 2400);
  await fullSpeed(AUTO ? 11500 : Infinity, [
    [2400, () => caption("Calibrated probabilities, <b>not generated text</b>", "", 3000)],
    [5800, () => caption(`Every question in <b>one engine pass</b>`, `X-SparkScore-Passes: 1 on every ticket so far`, 3000)],
    [9000, () => caption("Runs locally: <b>tickets never leave the box</b>", "one NVIDIA DGX Spark on a desk", 2500)],
  ]);
  caption(`<b>${stats.decisions}</b> typed decisions · p50 <b>${ms0(stats.p(0.5))} ms</b> per ticket`, `${stats.server.length} tickets · 4 questions each · server time, one DGX Spark`, 0);
  await sleep(3600);
  done();
}

// ===================================================== 2. Agent Guardrail
async function sceneGuard() {
  title("Agent Guardrail");
  root.className = "guard";
  const D = DEMO.guard;
  const H = hud([
    ["checks", "Tool calls checked", "var(--cyan)", ""],
    ["allowed", "Allowed", "var(--green)", ""],
    ["blocked", "Blocked", "var(--red)", ""],
    ["p50", "Check p50", "var(--violet)", ""],
    ["passes", "Passes / check", "var(--pink)", "measured"],
  ]);
  const term = el("div", "panel term", `<div class="term-h"><i></i><i></i><i></i><span>orbit — autonomous ops agent (simulated)</span></div><div class="term-b"><div class="lines"></div></div>`);
  const lines = term.querySelector(".lines");
  root.append(term);

  const rules = D.policy.split("\n").filter((l) => /^\d\./.test(l));
  const pol = el("div", "panel policy", `<div class="eyebrow">Written policy · in the state</div>`);
  const ruleEls = {};
  rules.forEach((r, k) => {
    const e = el("div", "rule", `<b>${k + 1}</b>${esc(r.replace(/^\d\.\s*/, ""))}`);
    ruleEls[`r${k + 1}`] = e;
    pol.append(e);
  });
  root.append(pol);

  const ver = el("div", "panel verdict", `<div class="insp-head"><span class="eyebrow">SparkScore verdict</span><span class="ms" id="vms">— ms</span></div>`);
  const vrow = el("div", "vrow");
  const vring = ring("ring", 78, 14, "p(violation)");
  const side = el("div", "vside");
  const stamp = el("div", "stamp wait", "WAITING");
  const riskBox = el("div", "", `<div class="lbl">Risk <span class="qtype">score</span></div>`);
  const risk = levels(D.questions.risk.criteria, ["#5d6890", "var(--cyan)", "var(--amber)", "var(--red)"]);
  riskBox.append(risk);
  const catBox = el("div", "", `<div class="lbl">Category <span class="qtype">choice</span></div>`);
  const cats = el("div", "cats");
  const catEls = {};
  for (const k of Object.keys(D.questions.category.criteria)) { catEls[k] = el("div", "cat", `<span>${k}</span>`); cats.append(catEls[k]); }
  catBox.append(cats);
  side.append(stamp, riskBox, catBox);
  vrow.append(vring, side);
  ver.append(vrow);
  root.append(ver);

  const stats = new Stats();
  let allowed = 0, blocked = 0;
  function line(cls, html) {
    const e = el("div", `ln ${cls}`, html);
    lines.append(e);
    while (lines.children.length > 40) lines.firstChild.remove();
    return e;
  }
  async function typeLine(call) {
    const e = line("prompt", `<span class="who">orbit ▸ </span><span class="tool"></span><span class="rest"></span><span class="caret"></span>`);
    const tool = e.querySelector(".tool"), rest = e.querySelector(".rest");
    const full = `(${call.args})`;
    for (const ch of call.tool) { tool.textContent += ch; await sleep(11); }
    for (let k = 0; k < full.length; k += 2) { rest.textContent = full.slice(0, k + 2); await sleep(13); }
    e.querySelector(".caret").remove();
  }

  caption("Every tool call is checked <b>before it runs</b>", "violates the policy? (noul) · risk (score) · rule (choice) · category (choice)", 5200);
  let firstBlock = true, midCap = false;
  const t0 = performance.now();
  for (let k = 0; k < D.calls.length; k++) {
    const call = D.calls[k];
    await typeLine(call);
    const chk = line("check", "  ⧗ SparkScore policy check …");
    stamp.className = "stamp wait"; stamp.textContent = "CHECKING";
    ver.classList.add("scan");
    const r = await systemOne("guard", D.state(call), D.questions);
    ver.classList.remove("scan");
    stats.add(r);
    const a = r.answers;
    const p = a.violates.noul, block = p >= 0.5;
    const lvl = risk.set(a.risk.probabilities);
    vring.set(p, block ? "var(--red)" : "var(--green)");
    for (const [c, e] of Object.entries(catEls)) { e.style.setProperty("--p", a.category.probabilities[c]); e.classList.toggle("on", c === a.category.choice); }
    for (const [rk, e] of Object.entries(ruleEls)) { e.classList.toggle("hot", block && rk === a.rule.choice); e.classList.toggle("cool", !block && rk === a.rule.choice); }
    document.getElementById("vms").textContent = `${ms0(r.server)} ms`;
    const riskName = D.questions.risk.criteria[lvl];
    const ruleName = a.rule.choice === "none" ? "no rule" : `rule ${a.rule.choice.slice(1)}`;
    if (block) {
      blocked++;
      stamp.className = "stamp block"; stamp.textContent = "BLOCKED";
      chk.className = "ln bad";
      chk.textContent = `  ✕ BLOCKED   p(violation) ${f2(p)} · ${riskName} · ${ruleName} · ${ms0(r.server)} ms`;
      term.classList.remove("redflash"); void term.offsetWidth; term.classList.add("redflash");
    } else {
      allowed++;
      stamp.className = "stamp allow"; stamp.textContent = "ALLOWED";
      chk.className = "ln ok";
      chk.textContent = `  ✓ allowed   p(violation) ${f2(p)} · ${riskName} · ${a.category.choice} · ${ms0(r.server)} ms`;
      term.classList.remove("greenflash"); void term.offsetWidth; term.classList.add("greenflash");
    }
    H.checks(stats.server.length);
    H.allowed(allowed); H.blocked(blocked);
    H.p50(ms0(stats.p(0.5)), "ms");
    H.p50.sub(`round trip ${ms0(stats.prt(0.5))} ms`);
    H.passes(r.passes, `× ${r.questions} questions`);
    if (block) {
      await sleep(380);
      line("agent", "  orbit: action blocked by policy · asking a human to review");
      if (firstBlock) {
        firstBlock = false;
        caption(`Blocked with a <b>calibrated probability</b>: ${f2(p)}`, `${riskName} · ${ruleName} · ${ms0(r.server)} ms on the server`, 2700);
        await sleep(1500);
      }
      await sleep(1100);
    } else {
      await sleep(160);
      for (const o of call.out || []) { line("out", `  ${esc(o)}`); await sleep(60); }
      await sleep(240);
    }
    if (!midCap && performance.now() - t0 > 13500) {
      midCap = true;
      caption("Fast enough to sit <b>inside the agent loop</b>", `check p50 ${ms0(stats.p(0.5))} ms so far, live`, 3800);
    }
  }
  await sleep(300);
  caption(`<b>${stats.server.length}</b> tool calls checked · <b>${blocked}</b> blocked · p50 <b>${ms0(stats.p(0.5))} ms</b>`, "4 typed questions per check, one engine pass each", 0);
  await sleep(3600);
  done();
}

// ===================================================== 3. Live as you type
async function sceneTyping() {
  title("Live as you type");
  root.className = "typing";
  const D = DEMO.typing;
  const H = hud([
    ["updates", "Updates", "var(--cyan)", "≤ 1 in flight"],
    ["last", "Last update", "var(--green)", ""],
    ["p50", "Update p50", "var(--violet)", ""],
    ["chars", "Draft length", "var(--amber)", ""],
    ["passes", "Passes / update", "var(--pink)", "measured"],
  ]);
  const custMsg = D.context.split("\n")[1];
  const cust = el("div", "panel cust", `<div class="eyebrow">Incoming</div><div class="from">Dana Whitcombe<span>monthly plan</span></div><p>${esc(custMsg)}</p>`);
  const compose = el("div", "panel compose", `<div class="insp-head"><span class="eyebrow">Your reply · draft</span><span class="ms" id="tms">— ms</span></div><div class="draft"><span class="txt"></span><span class="caret"></span></div>
    <div class="sendbar"><span class="eyebrow" style="font-size:15px">Policy: refunds OK · no free months or discounts</span><span class="sendbtn" id="send">Keep writing</span></div>`);
  root.append(cust, compose);

  const rev = el("div", "panel review", `<div class="insp-head"><span class="eyebrow">Live review · 5 typed questions</span></div>`);
  const toneQ = el("div", "rv-q", `<div class="ql"><span>Tone <span class="qtype">choice</span></span></div>`);
  const toneColors = { warm: "var(--green)", neutral: "var(--cyan)", defensive: "var(--amber)", hostile: "var(--red)" };
  const toneBars = {};
  for (const [k, c] of Object.entries(toneColors)) { toneBars[k] = pbar(k, c); toneBars[k].style.setProperty("--lw", "140px"); toneQ.append(toneBars[k]); }
  function bigQ(label, color, alarm) {
    const q = el("div", "rv-q", `<div class="ql"><span>${label} <span class="qtype">noul</span></span></div><div class="big"><div class="t"><div class="f"></div></div><div class="n">—</div></div>`);
    const b = q.querySelector(".big");
    b.style.setProperty("--c", color);
    q.set = (p) => {
      b.style.setProperty("--v", p);
      b.querySelector(".n").textContent = f2(p);
      b.classList.toggle("alarm", alarm && p >= 0.5);
      b.style.setProperty("--c", alarm && p >= 0.5 ? "var(--red)" : color);
    };
    return q;
  }
  const escQ = bigQ("Will this make the customer angrier?", "var(--cyan)", true);
  const promQ = bigQ("Promises something the policy forbids?", "var(--cyan)", true);
  const apoQ = bigQ("Apologizes?", "var(--green)", false);
  const resQ = el("div", "rv-q", `<div class="ql"><span>Resolves the problem? <span class="qtype">score</span></span></div>`);
  const res = levels(D.questions.resolves.criteria, ["#5d6890", "var(--amber)", "var(--cyan)", "var(--green)"]);
  resQ.append(res);
  const spark = el("div", "spark", `<div class="row"><span>server ms per update</span><span id="sp"></span></div><svg viewBox="0 0 750 56" preserveAspectRatio="none"><polyline fill="none" stroke="var(--cyan)" stroke-width="2.5" points=""/></svg>`);
  rev.append(toneQ, escQ, promQ, apoQ, resQ);
  compose.append(spark);
  root.append(rev);

  const txt = compose.querySelector(".txt");
  const stats = new Stats();
  let text = "", inflight = false, dirty = false, timer = null;
  function changed() {
    txt.textContent = text;
    H.chars(text.length, "chars");
    dirty = true;
    if (!inflight && !timer) timer = setTimeout(send, 40); // coalesce keystrokes, never stack requests
  }
  async function send() {
    timer = null;
    if (inflight || !dirty) return;
    inflight = true; dirty = false;
    const snapshot = text;
    const r = await systemOne("typing", D.state(snapshot), D.questions);
    inflight = false;
    stats.add(r);
    const a = r.answers;
    for (const [k, b] of Object.entries(toneBars)) b.set(a.tone.probabilities[k], k === a.tone.choice);
    escQ.set(a.escalate.noul);
    promQ.set(a.promise.noul);
    apoQ.set(a.apology.noul);
    const lvl = res.set(a.resolves.probabilities);
    const btn = document.getElementById("send");
    if (a.escalate.noul >= 0.5 && snapshot.length > 12) { btn.className = "sendbtn warn"; btn.textContent = "Likely to escalate"; }
    else if (a.promise.noul >= 0.5) { btn.className = "sendbtn warn"; btn.textContent = "Breaks support policy"; }
    else if (lvl >= 2) { btn.className = "sendbtn ok"; btn.textContent = "Ready to send ✓"; }
    else { btn.className = "sendbtn"; btn.textContent = "Keep writing"; }
    document.getElementById("tms").textContent = `${ms0(r.server)} ms`;
    H.updates(stats.server.length);
    H.last(ms0(r.server), "ms");
    H.last.sub(`round trip ${ms0(r.rt)} ms`);
    H.p50(ms0(stats.p(0.5)), "ms");
    H.passes(r.passes, `× ${r.questions} questions`);
    const pts = stats.server.slice(-60);
    const lo = 0, hi = Math.max(400, ...pts);
    spark.querySelector("polyline").setAttribute("points", pts.map((v, k) => `${(k / 59) * 750},${56 - ((v - lo) / (hi - lo)) * 52}`).join(" "));
    spark.querySelector("#sp").textContent = `min ${ms0(Math.min(...stats.server))} · p50 ${ms0(stats.p(0.5))} · max ${ms0(Math.max(...stats.server))}`;
    if (dirty) send();
  }

  changed();
  await sleep(600);
  const caps = {
    0: () => caption("Decisions update <b>as you type</b>", "tone · will it escalate · policy promise · apology · resolution", 4200),
    2: () => caption("Flags a reply that <b>will escalate</b>", "", 2600),
    7: () => caption("Catches a promise <b>the policy forbids</b>", "", 2600),
  };
  for (let k = 0; k < D.script.length; k++) {
    const op = D.script[k];
    if (caps[k]) caps[k]();
    if (op.type) {
      for (const ch of op.type) {
        text += ch; changed();
        await sleep(/[.,!?]/.test(ch) ? 150 : ch === " " ? 55 : 34 + Math.random() * 22);
      }
    } else if (op.del) {
      for (let n = 0; n < op.del; n++) { text = text.slice(0, -1); changed(); await sleep(11); }
    } else if (op.pause) await sleep(op.pause);
  }
  while (inflight || dirty) await sleep(50);
  caption(`Feedback in <b>${ms0(stats.p(0.5))} ms</b> per update`, `server p50 over ${stats.server.length} live updates · at most one request in flight`, 0);
  await sleep(3600);
  done();
}

// ============================================ 4. Many questions, one pass
async function sceneClaims() {
  title("Many questions, one pass");
  root.className = "claims";
  const D = DEMO.claims;
  const H = hud([
    ["n", "Decisions", "var(--cyan)", "one request"],
    ["passes", "Engine passes", "var(--pink)", "measured"],
    ["ms", "Server time", "var(--green)", ""],
    ["alone", "1 question alone", "var(--violet)", "same file"],
    ["tok", "Input tokens", "var(--amber)", ""],
  ]);
  const doc = el("div", "panel doc", `<div class="insp-head"><span class="eyebrow" id="dtitle"></span></div><pre></pre>`);
  root.append(doc);
  const UI = { // label, colour when the answer is yes
    claim_type: ["Type of loss", null, { water: "Water", fire: "Fire", theft: "Theft", storm: "Storm", liability: "Liability" }], covered: ["Cause of loss covered?", "var(--green)"], on_time: ["Reported within 14 days?", "var(--green)"],
    within_limit: ["Within the coverage limit?", "var(--green)"], severity: ["Damage severity"], injury: ["Anyone injured?", "var(--amber)"],
    fraud: ["Fraud red flags?", "var(--red)"], docs: ["Docs complete enough to pay?", "var(--green)"], habitable: ["Home still livable?", "var(--green)"],
    third_party: ["Another party liable?", "var(--amber)"], sentiment: ["Policyholder tone", null, { calm: "Calm", distressed: "Distressed", angry: "Angry" }],
    route: ["Route the claim", null, { fast_track: "Fast-track", adjuster: "Adjuster visit", investigate: "Investigate" }],
  };
  const grid = el("div", "qgrid");
  grid.style.top = "132px"; grid.style.height = "672px";
  const tiles = {};
  for (const [k, q] of Object.entries(D.questions)) {
    const t = el("div", "qt", `<div class="ql">${UI[k][0]}</div><span class="qtype">${q.type}</span>`);
    if (q.type === "noul") {
      t.r = ring("mini-ring ring", 42, 9, "p(yes)");
      t.yn = el("div", "yn", "");
      t.append(t.r, t.yn);
    } else if (q.type === "choice") {
      t.win = el("div", "winner", "—");
      t.bars = el("div", "");
      t.append(t.win, t.bars);
    } else {
      t.win = el("div", "winner", "—");
      t.lv = levels(q.criteria.map(() => ""), ["var(--green)", "var(--cyan)", "var(--amber)", "var(--red)"]);
      t.append(t.win, t.lv, el("div", "lvscale", `<span>${esc(q.criteria[0])}</span><span>${esc(q.criteria[q.criteria.length - 1])}</span>`));
    }
    tiles[k] = t;
    grid.append(t);
  }
  root.append(grid);

  const label = (k, key) => {
    const c = D.questions[k].criteria;
    return Array.isArray(c) ? c[Number(key)] : (UI[k][2] || c)[key];
  };
  function show(r) {
    for (const [k, a] of Object.entries(r.answers)) {
      const t = tiles[k];
      let color = "var(--cyan)";
      if (a.type === "noul") {
        const yes = a.noul >= 0.5;
        color = yes ? UI[k][1] : "#7d88b3";
        t.r.set(a.noul, color);
        t.yn.textContent = yes ? "YES" : "NO";
        t.yn.style.color = color;
      } else if (a.type === "choice") {
        color = "var(--violet)";
        t.win.textContent = label(k, a.choice);
        t.bars.replaceChildren(...Object.entries(a.probabilities).sort((x, y) => y[1] - x[1]).slice(0, 3).map(([key, p]) => {
          const b = pbar(label(k, key), "var(--violet)");
          b.set(p, key === a.choice);
          return b;
        }));
      } else {
        const lvl = t.lv.set(a.probabilities);
        color = ["var(--green)", "var(--cyan)", "var(--amber)", "var(--red)"][lvl];
        t.win.innerHTML = `${esc(label(k, lvl))} <small>${a.score.toFixed(2)}</small>`;
      }
      t.style.setProperty("--c", color);
      t.classList.remove("pop"); void t.offsetWidth; t.classList.add("pop");
    }
  }

  function clearTiles() {
    for (const t of Object.values(tiles)) {
      t.classList.add("pending");
      t.style.removeProperty("--c");
      if (t.r) { t.r.set(0, "var(--faint)"); t.r.querySelector("b").textContent = "—"; t.yn.textContent = ""; }
      if (t.bars) t.bars.replaceChildren();
      if (t.lv) t.lv.set({});
      if (t.win) t.win.textContent = "—";
    }
    for (const k of ["n", "passes", "ms", "alone", "tok"]) H[k]("—");
    H.ms.sub("");
  }

  const stats = new Stats();
  const first = Object.keys(D.questions)[0];
  function showDoc(d) {
    document.getElementById("dtitle").textContent = d.title;
    const pre = doc.querySelector("pre");
    const [head, ...rest] = d.text.split("\n");
    pre.innerHTML = `<span class="hd">${esc(head)}</span>\n${esc(rest.join("\n"))}`;
    doc.classList.remove("swap"); void doc.offsetWidth; doc.classList.add("swap");
  }
  async function run(d) {
    showDoc(d);
    clearTiles(); // the old file's answers never sit next to the new file
    await sleep(900);
    doc.classList.add("scan");
    const one = await systemOne("claims-1q", D.state(d), { [first]: D.questions[first] });
    H.alone(ms0(one.server), "ms");
    const r = await systemOne("claims", D.state(d), D.questions);
    doc.classList.remove("scan");
    grid.querySelectorAll(".qt").forEach((t) => t.classList.remove("pending"));
    stats.add(r);
    show(r);
    H.n(r.questions);
    H.passes(r.passes);
    H.ms(ms0(r.server), "ms");
    H.ms.sub(`round trip ${ms0(r.rt)} ms`);
    H.tok(r.tokens.toLocaleString("en-US"));
    return [r, one];
  }

  const [r, one] = await run(D.docs[0]);
  caption(`<b>${r.questions} typed decisions</b> in one engine pass`, `${ms0(r.server)} ms on the server · the same file with 1 question took ${ms0(one.server)} ms`, 4200);
  await sleep(4700);
  caption("Calibrated, so you choose <b>your own thresholds</b>", "noul · choice · score, each a probability distribution", 3200);
  await sleep(3700);

  // Full speed: the rest of the files back to back, one request in flight. A
  // file and its answers appear together, once its pass is done.
  caption("Now at <b>full speed</b>: a new claim file every half second", "12 questions per file, one request in flight, live", 0);
  H.alone.sub("first file");
  const fast = { files: 0, decisions: 0, server: [] };
  const fastStart = performance.now();
  const fastFor = AUTO ? 7500 : Infinity;
  for (let i = 1; performance.now() - fastStart < fastFor; i = i % (D.docs.length - 1) + 1) {
    const d = D.docs[i];
    const f = await systemOne("claims", D.state(d), D.questions);
    showDoc(d);
    grid.querySelectorAll(".qt").forEach((t) => t.classList.remove("pending"));
    show(f);
    fast.files++; fast.decisions += f.questions; fast.server.push(f.server);
    H.n(fast.decisions.toLocaleString("en-US"));
    H.n.sub(`${fast.files} files`);
    H.passes(f.passes);
    H.ms(ms0(f.server), "ms");
    H.ms.sub(`round trip ${ms0(f.rt)} ms`);
    H.tok(f.tokens.toLocaleString("en-US"));
    await sleep(16);
  }
  const secs = (performance.now() - fastStart) / 1000;
  caption(`<b>${fast.files} claim files</b>, <b>${fast.decisions} typed decisions</b> in ${secs.toFixed(1)} s`, `one engine pass per file · p50 ${ms0(Stats.pct(fast.server, 0.5))} ms per file · one DGX Spark`, 0);
  await sleep(4200);
  caption("Runs locally: <b>the files never leave the box</b>", "d3-ultra 27B · NVFP4 · one DGX Spark", 0);
  await sleep(3400);
  done();
}

// ================================================== 5. Knowledge packs
async function scenePacks() {
  title("Knowledge packs");
  root.className = "packs";
  const D = DEMO.packs;
  const H = hud([
    ["mode", "Mode", "var(--violet)", ""],
    ["ms", "Server time", "var(--green)", "this ticket"],
    ["n", "Tickets", "var(--cyan)", ""],
    ["rules", "Rules followed", "var(--amber)", "handbook overrides"],
    ["tok", "Handbook tokens", "var(--pink)", ""],
  ]);
  const text = await (await fetch(D.handbook)).text();
  const hb = el("div", "panel hb", `<div class="insp-head"><span class="eyebrow">${esc(D.title)}</span><span class="hbchip" id="hbchip"></span></div><pre></pre><div class="packbadge" id="pb"></div>`);
  const pre = hb.querySelector("pre");
  pre.innerHTML = text.split("\n\n").map((para) => {
    const i = para.indexOf("Override:");
    return i < 0 ? esc(para) : `${esc(para.slice(0, i))}<span class="rule">${esc(para.slice(i))}</span>`;
  }).join("\n\n");
  root.append(hb);

  const tk = el("div", "panel tk", `<div class="insp-head"><span class="eyebrow" id="tke">Ticket</span><span class="ms" id="tkms">— ms</span></div>
    <div class="who" id="tkwho"></div><div class="ttext" id="tktext"></div><div class="timer" id="timer"></div>`);
  const rows = {};
  const qrow = (key, label, type) => {
    const r = el("div", "qblock arow", `<div class="ql">${label} <span class="qtype">${type}</span><span class="rbadge"></span><span class="mark"></span></div>`);
    rows[key] = r; tk.append(r); return r;
  };
  const teamBars = {};
  const tr = qrow("team", "Which team?", "choice");
  for (const [k, name] of Object.entries(D.questions.team.criteria)) { teamBars[k] = pbar(name, "var(--cyan)"); teamBars[k].style.setProperty("--lw", "190px"); tr.append(teamBars[k]); }
  const urg = levels(D.questions.urgency.criteria, URG_COLORS); qrow("urgency", "Urgency", "score").append(urg);
  const legal = pbar("Legal review", "var(--pink)"); legal.style.setProperty("--lw", "190px"); qrow("legal", "Needs legal review?", "noul").append(legal);
  root.append(tk);
  const $ = (id) => root.querySelector(id);

  const truth = { team: (a) => a.team.choice, urgency: (a) => argmax(D.questions.urgency.criteria.map((_, i) => a.urgency.probabilities[String(i)] || 0)), legal: (a) => (a.legal.noul >= 0.5 ? 1 : 0) };
  const tally = { ok: 0, n: 0, tickets: 0 };
  function showTicket(t, label) {
    $("#tke").textContent = label;
    $("#tkwho").innerHTML = `${esc(t.name)} · ${esc(t.company)}<span class="plan">customer tier: ${esc(t.tier)}</span>`;
    $("#tktext").textContent = `“${t.text}”`;
    for (const [q, r] of Object.entries(rows)) {
      const rule = t.rules.find((x) => D.ruleQuestion[x] === q);
      r.querySelector(".rbadge").textContent = rule ? `${D.rules[rule][0]} ${D.rules[rule][1]} ${D.rules[rule][2]}` : "";
      r.querySelector(".mark").textContent = "";
      r.classList.remove("ok", "bad");
    }
  }
  function showAnswers(t, r, count = true) {
    const a = r.answers;
    for (const [k, b] of Object.entries(teamBars)) b.set(a.team.probabilities[k], k === a.team.choice);
    urg.set(a.urgency.probabilities);
    legal.set(a.legal.noul, a.legal.noul >= 0.5);
    $("#tkms").textContent = `${ms0(r.server)} ms`;
    let ok = 0;
    for (const rule of t.rules) {
      const q = D.ruleQuestion[rule], good = truth[q](a) === t.gold[q];
      ok += good;
      const row = rows[q];
      row.classList.add(good ? "ok" : "bad");
      row.querySelector(".mark").textContent = good ? "✓" : "✗";
    }
    if (count) { tally.ok += ok; tally.n += t.rules.length; tally.tickets += 1; }
    return ok;
  }
  function timer(on) {
    const e = $("#timer");
    if (!on) { cancelAnimationFrame(e.raf); return; }
    const t0 = performance.now();
    const tick = () => { e.textContent = `${((performance.now() - t0) / 1000).toFixed(2)} s`; e.raf = requestAnimationFrame(tick); };
    e.classList.add("on"); tick();
  }
  async function scroll(ms) {
    const t0 = performance.now(), max = pre.scrollHeight - pre.clientHeight;
    while (performance.now() - t0 < ms) { pre.scrollTop = max * ((performance.now() - t0) / ms); await sleep(16); }
    pre.scrollTop = 0;
  }

  // 1. The handbook.
  const hero = D.hero;
  H.mode("—"); H.n("0"); H.rules("—"); H.tok("—");
  $("#hbchip").textContent = `${Math.round(text.length / 1000)}K characters · 7 override rules`;
  caption("A company's <b>support handbook</b>: 58,000 characters, 7 house rules", "routing, urgency and legal overrides, spread through ordinary sections", 0);
  await scroll(AUTO ? 5200 : 3500);

  // 2. Without it.
  showTicket(hero, "Ticket");
  caption("Without the handbook, the model <b>can't know the house rules</b>", "one ticket, three questions: team · urgency · legal", 0);
  H.mode("No handbook");
  const none = await systemOne("packs-none", D.state(hero), D.questions);
  showAnswers(hero, none, false);
  H.ms(ms0(none.server), "ms");
  await sleep(3600);

  // 3. In every request.
  showTicket(hero, "Ticket");
  caption("<b>Today</b>: the handbook goes in every request", "d3 reads it together with each ticket, so no prefix cache can reuse it: 12K tokens recomputed per ticket", 0);
  H.mode("Today: in request");
  hb.classList.add("scan");
  timer(true);
  const inReq = await systemOne("packs-in-request", `${text}\n\n${D.state(hero)}`, D.questions);
  timer(false); hb.classList.remove("scan");
  const todayOk = showAnswers(hero, inReq, false);
  H.ms(ms0(inReq.server), "ms");
  H.tok(inReq.tokens.toLocaleString("en-US"));
  await sleep(3800);

  // 4. A knowledge pack, built once.
  caption("Or register it once as a <b>knowledge pack</b>", "PUT /v1/knowledge/acme-handbook · the engine keeps its attention keys and recurrent state", 0);
  const t0 = performance.now();
  const put = await fetch(`/v1/knowledge/${D.name}`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ text }) });
  const reg = await put.json();
  if (!put.ok) throw new Error(`knowledge ${put.status}: ${JSON.stringify(reg.detail)}`);
  const built = performance.now() - t0;
  $("#pb").innerHTML = `<b>KNOWLEDGE PACK</b> · ${reg.pack_tokens.toLocaleString("en-US")} tokens · ${reg.pack === "built" ? `built in ${(built / 1000).toFixed(1)} s, once` : "held by the engine"}`;
  hb.classList.add("packed");
  H.tok(reg.pack_tokens.toLocaleString("en-US"));
  H.tok.sub("read in place");
  $("#hbchip").textContent = `${reg.pack_tokens.toLocaleString("en-US")} tokens · 7 override rules`;
  await sleep(3400);

  // 5. Every ticket on the pack, for as long as one in-request ticket took.
  H.mode("Knowledge pack");
  showTicket(hero, "Ticket");
  const first = await systemOne("packs", D.state(hero), D.questions, { knowledge_name: D.name });
  showAnswers(hero, first);
  H.ms(ms0(first.server), "ms"); H.n(1); H.rules(`${tally.ok}/${tally.n}`);
  caption(`Same answers in <b>${ms0(first.server)} ms</b>, not ${(inReq.server / 1000).toFixed(1)} s`, "the handbook is read in place, never recomputed", 0);
  await sleep(3600);
  caption(`Now: as many tickets as fit in <b>${(inReq.server / 1000).toFixed(1)} s</b>, the time one took before`, "live, one request in flight", 0);
  const budget = inReq.rt, start = performance.now(), times = [first.server];
  let i = 0;
  timer(true);
  while (performance.now() - start < budget) {
    const t = D.stream[i++ % D.stream.length];
    const r = await systemOne("packs", D.state(t), D.questions, { knowledge_name: D.name });
    showTicket(t, "Ticket");
    showAnswers(t, r);
    times.push(r.server);
    H.ms(ms0(r.server), "ms"); H.n(tally.tickets); H.rules(`${tally.ok}/${tally.n}`);
    await sleep(16);
  }
  timer(false);
  const p50 = Stats.pct(times, 0.5);
  const lane = (cls, name, how, per, n, ok, of) => `<div class="lane ${cls}"><div class="lname">${name}</div><div class="lhow">${how}</div>
    <div class="lbig">${per}<small>per ticket</small></div><div class="lstat"><b>${n}</b> ticket${n === 1 ? "" : "s"} in ${(inReq.server / 1000).toFixed(1)} s</div>
    <div class="lstat"><b>${ok}/${of}</b> house rules followed</div></div>`;
  const race = el("div", "panel race", `<div class="eyebrow">Same tickets · same engine · same DGX Spark</div><div class="lanes">
    ${lane("today", "Today", "handbook in every request", `${(inReq.server / 1000).toFixed(2)} s`, 1, todayOk, hero.rules.length)}
    ${lane("ours", "Knowledge pack", "handbook read once", `${ms0(p50)} ms`, tally.tickets, tally.ok, tally.n)}</div>`);
  root.append(race);
  await sleep(50); race.classList.add("on");
  caption(`<b>${Math.round(inReq.server / p50)}× faster</b> per ticket, with the same house rules followed`, `12K-token handbook · p50 ${ms0(p50)} ms on a pack vs ${ms0(inReq.server)} ms in the request`, 0);
  await sleep(6400);
  caption("<b>Knowledge packs</b>: your rulebook, read once", "knowledge_name in /v1/systemone · Atlas --score-packs", 0);
  await sleep(3600);
  done();
}

// ================================================================= titles
const CARDS = {
  intro: { kicker: "Live demo", h1: "SparkScore", h2: "Typed decisions in <b>~150 ms</b> on one DGX Spark",
    facts: ["state in → calibrated probabilities out", "nothing is generated", "one forward pass"],
    foot: "~150 ms: JevBench public p50 (146 ms). The demos use longer prompts and show their own live timings." },
  s1: { kicker: "01", h1: "Inbox Firehose", h2: "Route, rank and flag every ticket", facts: ["choice · score · noul · noul", "one call per ticket"] },
  s2: { kicker: "02", h1: "Agent Guardrail", h2: "Check every tool call <b>before it runs</b>", facts: ["against a written policy", "inside the agent loop"] },
  s3: { kicker: "03", h1: "Live as you type", h2: "Decisions that keep up with the keyboard", facts: ["5 questions per update", "one request in flight"] },
  s4: { kicker: "04", h1: "Many questions, one pass", h2: "12 typed decisions per document", facts: ["one request per file", "one engine pass"] },
  packs: { kicker: "Knowledge packs", h1: "Your rulebook, read once", h2: "12K-token handbook: <b>191 ms</b> a ticket, not 5.4 s",
    facts: ["same answers as the handbook in the request", "PUT /v1/knowledge/NAME · knowledge_name", "Atlas --score-packs · one DGX Spark"],
    foot: "Measured on one DGX Spark (results/2026-11-08-knowledge-packs). The scene's own timings are live." },
  s5: { kicker: "05", h1: "Knowledge packs", h2: "A 12K-token handbook, <b>read once</b>", facts: ["register a document by name", "every request reads it in place", "same answers as in the request"] },
  outro: { kicker: "SparkScore", h1: "System One, local", h2: "JevBench public <b>90.9%</b> at p50 <b>146 ms</b>",
    facts: ["d3-ultra 27B (vllm-sr) · NVFP4 W4A4", "Atlas engine · one NVIDIA DGX Spark", "d3's BF16 runtime, same box: 91.3% at 633 ms", "runs locally: the data stays on the box"] },
};
async function sceneTitle() {
  const c = CARDS[P.get("card") || "intro"];
  document.querySelector(".topbar").style.display = "none";
  root.innerHTML = `<div class="titlecard"><div class="kicker">${c.kicker}</div><h1>${c.h1}</h1><h2>${c.h2}</h2><div class="facts">${c.facts.map((f) => `<span>${f}</span>`).join("")}</div>${c.foot ? `<div class="foot">${c.foot}</div>` : ""}</div>`;
  document.querySelector(".note").textContent = "Synthetic demo data · fictional companies and people";
  await sleep(Number(P.get("hold") || 3500));
  done();
}

function sceneHome() {
  title("Live demo");
  root.innerHTML = `<div class="home">
    <a href="?scene=inbox"><em>01</em><b>Inbox Firehose</b><span>Support tickets routed, ranked and flagged: four typed questions per ticket, one engine pass.</span></a>
    <a href="?scene=guard"><em>02</em><b>Agent Guardrail</b><span>An agent's tool calls checked against a written policy before they run.</span></a>
    <a href="?scene=typing"><em>03</em><b>Live as you type</b><span>A draft reply reviewed on every keystroke burst, one request in flight.</span></a>
    <a href="?scene=claims"><em>04</em><b>Many questions, one pass</b><span>Twelve typed decisions about one insurance claim in a single request.</span></a>
    <a href="?scene=packs"><em>05</em><b>Knowledge packs</b><span>A 12K-token handbook registered once; every ticket follows its house rules at full speed.</span></a>
  </div>`;
}

const SCENES = { inbox: sceneInbox, guard: sceneGuard, typing: sceneTyping, claims: sceneClaims, packs: scenePacks, title: sceneTitle, home: sceneHome };
(async () => {
  await document.fonts.ready;
  if (SCENE !== "title") await health();
  window.__ready = true;
  if (AUTO) await sleep(Number(P.get("lead") || 600));
  try {
    await (SCENES[SCENE] || sceneHome)();
  } catch (error) {
    console.error(error);
    caption("Endpoint error", String(error.message || error), 0);
    window.__demoError = String(error);
    done();
  }
})();
