"""Render a run directory into a self-contained HTML report.

    python scripts/render_replay_page.py runs/agent-light-bulb-<stamp> [out.html]

Accepts a run directory (replay.json plus optional trace-<slot>.jsonl files)
or a bare replay.json path. The page shows the canvas on the left, the
message board on the right, small score / token / time charts along the
bottom (x axis is turns), and a collapsible full trace per agent.

The page is content-only HTML (no doctype/html/head/body wrapper) so it can
be published directly as a Claude artifact; browsers also render it
standalone.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

TEMPLATE = """<title>__TITLE__</title>
<style>
:root {
  --paper: #FAF7F0;
  --card: #FFFFFF;
  --grid: #E9E3D6;
  --line: #D8D2C4;
  --ink: #26221A;
  --muted: #7A7466;
  --accent: #B45309;
  --good: #15803D;
  --bad: #B91C1C;
  --tint: #FBF3E4;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #161512;
    --card: #201E19;
    --grid: #2E2B24;
    --line: #3A362D;
    --ink: #EAE6DC;
    --muted: #948D7D;
    --accent: #F59E0B;
    --good: #4ADE80;
    --bad: #F87171;
    --tint: #2A2620;
  }
}
:root[data-theme="dark"] {
  --paper: #161512;
  --card: #201E19;
  --grid: #2E2B24;
  --line: #3A362D;
  --ink: #EAE6DC;
  --muted: #948D7D;
  --accent: #F59E0B;
  --good: #4ADE80;
  --bad: #F87171;
  --tint: #2A2620;
}
body {
  background: var(--paper);
  color: var(--ink);
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  margin: 0;
  padding: 24px 16px 64px;
  font-variant-numeric: tabular-nums;
}
.wrap { max-width: 1160px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
.eyebrow { text-transform: uppercase; letter-spacing: 0.14em; font-size: 12px; color: var(--muted); }
h1 { font-size: 22px; font-weight: 700; margin: 4px 0 0; letter-spacing: 0.02em; text-wrap: balance; }
.statrow { display: flex; flex-wrap: wrap; gap: 12px 28px; align-items: baseline; }
.stat { display: flex; flex-direction: column; gap: 2px; }
.stat .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); }
.stat .v { font-size: 18px; font-weight: 700; }
.stat .v.pass { color: var(--good); }
.stat .v.fail { color: var(--bad); }
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 16px 18px 12px;
  min-width: 0;
}
.card h2 {
  font-size: 13px; font-weight: 700; letter-spacing: 0.08em;
  margin: 0 0 12px; text-transform: uppercase;
  border-bottom: 1px solid var(--line); padding-bottom: 10px;
}
.top { display: grid; grid-template-columns: minmax(380px, auto) minmax(300px, 1fr); gap: 18px; align-items: stretch; }
@media (max-width: 900px) { .top { grid-template-columns: 1fr; } }
.gridbox { overflow-x: auto; }
#board {
  display: grid;
  gap: 1px;
  background: var(--grid);
  border: 1px solid var(--grid);
  width: max-content;
  margin: 0 auto;
}
#board .c {
  width: 20px; height: 20px; background: var(--card);
  display: flex; align-items: center; justify-content: center;
  font-size: 9px; font-weight: 700; line-height: 1; user-select: none;
}
#board .c.changed { box-shadow: inset 0 0 0 2.5px var(--ink); }
.cardfoot {
  display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  border-top: 1px solid var(--line);
  margin-top: 12px; padding-top: 10px;
  font-size: 12px; color: var(--muted);
}
.controls { display: flex; align-items: center; gap: 10px; margin-top: 14px; }
.controls button {
  font: inherit; color: var(--ink); background: var(--card);
  border: 1px solid var(--line); border-radius: 4px;
  padding: 4px 12px; cursor: pointer;
}
.controls button:hover { border-color: var(--muted); }
.controls button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.controls input[type="range"] { flex: 1; accent-color: var(--accent); }
.turnlabel { min-width: 76px; text-align: right; font-size: 13px; }
.boardmsgs { display: flex; flex-direction: column; }
.msgs { flex: 1; min-height: 200px; max-height: 560px; overflow-y: auto; display: flex; flex-direction: column; gap: 9px; font-size: 12.5px; line-height: 1.5; }
.msg { display: flex; gap: 9px; align-items: baseline; padding: 2px 6px; border-radius: 4px; }
.msg.now { background: var(--tint); }
.chip { flex: none; width: 10px; height: 10px; border-radius: 2px; transform: translateY(1px); }
.msg .who { color: var(--muted); flex: none; }
.charts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
@media (max-width: 900px) { .charts { grid-template-columns: 1fr; } }
.chartcard h2 { margin-bottom: 8px; }
.chartcard svg { width: 100%; height: 96px; display: block; cursor: crosshair; }
.chartfoot { font-size: 12px; color: var(--muted); margin-top: 6px; min-height: 16px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 12.5px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.tip {
  position: fixed; pointer-events: none; z-index: 10;
  background: var(--card); border: 1px solid var(--line); border-radius: 4px;
  padding: 6px 9px; font-size: 12px; line-height: 1.5; color: var(--ink);
  box-shadow: 0 2px 8px rgba(0,0,0,0.12); max-width: 260px;
}
.traces { display: flex; flex-direction: column; gap: 10px; }
details.agent { background: var(--card); border: 1px solid var(--line); border-radius: 6px; }
details.agent > summary {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  cursor: pointer; padding: 12px 16px; font-size: 13px; font-weight: 700;
  list-style: none;
}
details.agent > summary::-webkit-details-marker { display: none; }
details.agent > summary::before { content: "\\25B8"; color: var(--muted); }
details.agent[open] > summary::before { content: "\\25BE"; }
details.agent > summary .meta { font-weight: 400; color: var(--muted); }
.trace { border-top: 1px solid var(--line); padding: 12px 16px; display: flex; flex-direction: column; gap: 14px; }
.tturn { border-left: 3px solid var(--grid); padding-left: 12px; }
.tturn .tthead { font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.ev { font-size: 12.5px; line-height: 1.55; margin: 4px 0; overflow-wrap: anywhere; }
.ev.think { color: var(--muted); font-style: italic; border-left: 2px solid var(--accent); padding-left: 9px; }
.evtag {
  display: inline-block; font-style: normal; font-size: 10px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent);
  margin-right: 6px;
}
.tturn.interview { border-left-color: var(--accent); }
.ev.tool { }
.ev.tool code { background: var(--paper); border: 1px solid var(--grid); border-radius: 3px; padding: 1px 5px; }
.ev.fail { color: var(--bad); }
details.res { margin: 4px 0; }
details.res summary { font-size: 12px; color: var(--muted); cursor: pointer; }
details.res pre { font-size: 11.5px; background: var(--paper); border: 1px solid var(--grid); border-radius: 4px; padding: 8px 10px; overflow-x: auto; white-space: pre-wrap; margin: 6px 0 0; }
</style>
<div class="wrap">
  <header>
    <div class="eyebrow" id="eyebrow">codrawing replay</div>
    <h1 id="title"></h1>
  </header>
  <div class="statrow" id="stats"></div>

  <div class="top">
    <section class="card">
      <h2 id="canvastitle">Canvas</h2>
      <div class="gridbox"><div id="board"></div></div>
      <div class="controls">
        <button id="prev" aria-label="previous turn">&#8249;</button>
        <button id="play">play</button>
        <button id="next" aria-label="next turn">&#8250;</button>
        <input type="range" id="scrub" min="0" value="0">
        <span class="turnlabel" id="turnlabel"></span>
      </div>
      <div class="cardfoot"><span id="lastaction"></span><span id="framestamp"></span></div>
    </section>
    <section class="card boardmsgs">
      <h2>Message board</h2>
      <div class="msgs" id="msgs"></div>
    </section>
  </div>

  <div class="charts">
    <section class="card chartcard">
      <h2>Score</h2>
      <svg id="score-svg" viewBox="0 0 400 96" preserveAspectRatio="none"></svg>
      <div class="chartfoot" id="score-foot"></div>
    </section>
    <section class="card chartcard" id="tokens-card">
      <h2>Output tokens per turn</h2>
      <svg id="tokens-svg" viewBox="0 0 400 96" preserveAspectRatio="none"></svg>
      <div class="chartfoot" id="tokens-foot"></div>
    </section>
    <section class="card chartcard" id="time-card">
      <h2>Seconds per turn</h2>
      <svg id="time-svg" viewBox="0 0 400 96" preserveAspectRatio="none"></svg>
      <div class="chartfoot" id="time-foot"></div>
    </section>
  </div>

  <div class="legend" id="legend"></div>

  <section class="card" id="interviews" style="display:none">
    <h2>Post-episode interviews</h2>
    <div id="interviews-body"></div>
  </section>

  <section class="traces" id="traces"></section>
</div>
<div id="tip" class="tip" hidden></div>
<script type="application/json" id="replay-data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("replay-data").textContent);
const frames = data.frames;
const results = data.results;
const traces = data.traces || {};
const W = frames[0].width, H = frames[0].height;
const SEATS = ["#EF4444", "#3B82F6", "#22C55E", "#F59E0B", "#A855F7"];
// CVD-validated stacking order for the token chart (green, blue, amber, red, purple).
const STACK_ORDER = [2, 1, 3, 0, 4];
const names = frames[0].player_names;
const seatCount = names.length;
const WHITE = "#FFFFFF";
const hasTraces = Object.keys(traces).length > 0;

function esc(v) { const d = document.createElement("div"); d.textContent = v; return d.innerHTML; }
function fmtTokens(n) { return n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n); }
function inkFor(hex) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return 0.299 * r + 0.587 * g + 0.114 * b > 140 ? "#26221A" : "#FFFFFF";
}

document.getElementById("eyebrow").textContent = "codrawing replay \\u00b7 " + (results.image_model || "");
document.getElementById("title").textContent =
  "Target: " + frames[0].target + " \\u00b7 " + seatCount + " agents \\u00b7 " + frames[0].max_turns + " turns";
document.getElementById("canvastitle").textContent = "Canvas (" + W + "\\u00d7" + H + ")";

function turnRecords(recs) { return recs.filter(r => (r.phase || "turn") === "turn"); }

// Older traces recorded the session-cumulative cost each turn; detect and use the last value.
function totalCost(recs) {
  const costs = recs.map(r => (r.usage && r.usage.cost_usd) || 0).filter(c => c > 0);
  if (!costs.length) return 0;
  const cumulative = costs.length >= 3 && costs.every((c, i) => i === 0 || c >= costs[i - 1] - 1e-9);
  return cumulative ? costs[costs.length - 1] : costs.reduce((a, b) => a + b, 0);
}

// Per-frame trace lookup: agent trace turn t produced the frame stamped t+1.
const traceByTurn = {};
for (const slot of Object.keys(traces)) {
  for (const rec of turnRecords(traces[slot])) {
    const frameTurn = rec.turn + 1;
    (traceByTurn[frameTurn] = traceByTurn[frameTurn] || {})[slot] = rec;
  }
}
function frameUsage(frameTurn) {
  const bySlot = traceByTurn[frameTurn] || {};
  let tokens = {}, seconds = 0;
  for (const slot of Object.keys(bySlot)) {
    tokens[slot] = (bySlot[slot].usage && bySlot[slot].usage.output_tokens) || 0;
    seconds = Math.max(seconds, bySlot[slot].wall_seconds || 0);
  }
  return { tokens: tokens, seconds: seconds, bySlot: bySlot };
}

const finalFb = results.final_image_model_feedback || {};
const bestScore = results.best_target_score !== undefined
  ? results.best_target_score
  : Math.max(...frames.map(fr => fr.image_model_feedback ? fr.image_model_feedback.target_score : 0));
let totalOut = 0, totalCostAll = 0, totalSeconds = 0;
for (const fr of frames) {
  const u = frameUsage(fr.turn);
  for (const s of Object.keys(u.tokens)) totalOut += u.tokens[s];
  totalSeconds += u.seconds;
}
for (const slot of Object.keys(traces)) totalCostAll += totalCost(traces[slot]);
const stats = [
  ["best score", (bestScore * 100).toFixed(1) + "%", ""],
  ["final score", finalFb.target_score !== undefined ? (finalFb.target_score * 100).toFixed(1) + "%" : "\\u2014", ""],
  ["threshold", finalFb.pass_threshold !== undefined ? (finalFb.pass_threshold * 100).toFixed(0) + "%" : "\\u2014", ""],
  ["evaluation", results.evaluation_passed ? "PASS" : "NOT PASSING", results.evaluation_passed ? "pass" : "fail"],
  ["turns", results.turns + "/" + (results.max_turns || frames[0].max_turns) +
    (results.ended_by_agents ? " \\u00b7 ended by agents" : ""), ""],
  ["accepted pixels", results.accepted_pixels.join(" / "), ""],
];
if (hasTraces) {
  stats.push(["output tokens", fmtTokens(totalOut), ""]);
  if (totalCostAll) stats.push(["cost (api equiv)", "$" + totalCostAll.toFixed(2), ""]);
  stats.push(["agent time", Math.round(totalSeconds / 60) + "m", ""]);
}
document.getElementById("stats").innerHTML = stats.map(([k, v, cls]) =>
  '<div class="stat"><span class="k">' + k + '</span><span class="v ' + cls + '">' + v + "</span></div>"
).join("");

document.getElementById("legend").innerHTML = names.map((n, i) =>
  '<span><span class="chip" style="background:' + SEATS[i] + '"></span>' + esc(n) + "</span>"
).join("");

const board = document.getElementById("board");
board.style.gridTemplateColumns = "repeat(" + W + ", 20px)";
const cells = [];
for (let i = 0; i < W * H; i++) {
  const d = document.createElement("div");
  d.className = "c";
  board.appendChild(d);
  cells.push(d);
}

const scrub = document.getElementById("scrub");
scrub.max = frames.length - 1;

function show(f) {
  scrub.value = f;
  const frame = frames[f];
  const prev = f > 0 ? frames[f - 1] : null;
  const changed = [];
  for (let i = 0; i < W * H; i++) {
    const color = frame.canvas[i];
    const owner = frame.owners ? frame.owners[i] : -1;
    cells[i].style.background = color === WHITE ? "" : color;
    if (color !== WHITE && owner >= 0) {
      cells[i].textContent = String(owner + 1);
      cells[i].style.color = inkFor(color);
    } else {
      cells[i].textContent = "";
    }
    const was = prev ? prev.canvas[i] : WHITE;
    const isChanged = was !== color;
    cells[i].classList.toggle("changed", isChanged);
    if (isChanged) changed.push(i);
  }
  document.getElementById("turnlabel").textContent = "T" + frame.turn + "/" + frame.max_turns;
  document.getElementById("framestamp").textContent = "frame " + (f + 1) + " of " + frames.length;

  const acts = changed.map(i => {
    const owner = frame.owners ? frame.owners[i] : null;
    const who = owner === null || owner === undefined || owner < 0 ? "?" : names[owner];
    const verb = frame.canvas[i] === WHITE ? "erased" : "painted";
    return who + " " + verb + " (" + (i % W) + "," + Math.floor(i / W) + ")";
  });
  document.getElementById("lastaction").textContent = acts.length ? acts.join(" \\u00b7 ") : "no accepted writes";

  const feed = [];
  for (let i = 0; i <= f; i++) {
    for (const m of frames[i].messages || []) {
      feed.push('<div class="msg' + (i === f ? " now" : "") + '">' +
        '<span class="chip" style="background:' + SEATS[m.slot] + '"></span>' +
        '<span class="who">T' + frames[i].turn + " " + esc(m.player) + "</span><span>" + esc(m.text) + "</span></div>");
    }
  }
  const msgs = document.getElementById("msgs");
  msgs.innerHTML = feed.join("") || '<span style="color:var(--muted)">no messages yet</span>';
  msgs.scrollTop = msgs.scrollHeight;

  drawScore(f);
  if (hasTraces) { drawTokens(f); drawTime(f); }
  const fb = frame.image_model_feedback;
  document.getElementById("score-foot").textContent = fb
    ? "T" + frame.turn + " \\u00b7 " + (fb.target_score * 100).toFixed(2) + "% \\u00b7 rank " +
      fb.target_rank + "/" + fb.label_count + " \\u00b7 top: " +
      fb.top_predictions.slice(0, 3).map(p => p.label + " " + (p.probability * 100).toFixed(0) + "%").join(", ")
    : "no classifier feedback";
  if (hasTraces) {
    const u = frameUsage(frame.turn);
    const out = Object.values(u.tokens).reduce((a, b) => a + b, 0);
    document.getElementById("tokens-foot").textContent = "T" + frame.turn + " \\u00b7 " + fmtTokens(out) + " tokens out";
    document.getElementById("time-foot").textContent =
      "T" + frame.turn + " \\u00b7 " + u.seconds.toFixed(1) + "s (slowest agent; turns are a barrier)";
  }
}

const PAD = { l: 6, r: 6, t: 8, b: 8 };
const PW = 400 - PAD.l - PAD.r, PH = 96 - PAD.t - PAD.b;
const xAt = i => PAD.l + (frames.length > 1 ? (i / (frames.length - 1)) * PW : PW / 2);
const slotX = i => PAD.l + (i / frames.length) * PW;
const slotW = Math.max(2, PW / frames.length - 2);

function drawScore(f) {
  const scores = frames.map(fr => fr.image_model_feedback ? fr.image_model_feedback.target_score : 0);
  const threshold = results.evaluation_threshold || 0;
  const max = Math.max(...scores, threshold, 0.01);
  const y = v => PAD.t + PH - (v / max) * PH;
  const path = scores.map((v, i) => (i ? "L" : "M") + xAt(i).toFixed(1) + " " + y(v).toFixed(1)).join(" ");
  const ty = y(threshold);
  document.getElementById("score-svg").innerHTML =
    '<line x1="' + PAD.l + '" y1="' + ty + '" x2="' + (400 - PAD.r) + '" y2="' + ty +
      '" stroke="var(--line)" stroke-dasharray="4 4" stroke-width="1"></line>' +
    '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2"></path>' +
    '<circle cx="' + xAt(f) + '" cy="' + y(scores[f]) + '" r="4" fill="var(--accent)"></circle>';
}

function drawTokens(f) {
  let max = 1;
  for (const fr of frames) {
    const u = frameUsage(fr.turn);
    max = Math.max(max, Object.values(u.tokens).reduce((a, b) => a + b, 0));
  }
  const parts = [];
  frames.forEach((fr, i) => {
    const u = frameUsage(fr.turn);
    let yTop = PAD.t + PH;
    for (const slot of STACK_ORDER) {
      const v = u.tokens[slot] || 0;
      if (!v) continue;
      const h = (v / max) * PH;
      yTop -= h;
      parts.push('<rect x="' + slotX(i).toFixed(1) + '" y="' + yTop.toFixed(1) +
        '" width="' + slotW.toFixed(1) + '" height="' + Math.max(0, h - 1).toFixed(1) +
        '" rx="1" fill="' + SEATS[slot] + '"' + (i === f ? "" : ' opacity="0.55"') + "></rect>");
    }
  });
  document.getElementById("tokens-svg").innerHTML = parts.join("");
}

function drawTime(f) {
  let max = 1;
  for (const fr of frames) max = Math.max(max, frameUsage(fr.turn).seconds);
  const parts = [];
  frames.forEach((fr, i) => {
    const v = frameUsage(fr.turn).seconds;
    if (!v) return;
    const h = (v / max) * PH;
    parts.push('<rect x="' + slotX(i).toFixed(1) + '" y="' + (PAD.t + PH - h).toFixed(1) +
      '" width="' + slotW.toFixed(1) + '" height="' + h.toFixed(1) +
      '" rx="1" fill="var(--muted)"' + (i === f ? "" : ' opacity="0.45"') + "></rect>");
  });
  document.getElementById("time-svg").innerHTML = parts.join("");
}

// Shared hover tooltip + click-to-seek on all three charts.
const tip = document.getElementById("tip");
function chartFrameIndex(svg, event) {
  const rect = svg.getBoundingClientRect();
  const frac = (event.clientX - rect.left) / rect.width;
  return Math.max(0, Math.min(frames.length - 1, Math.round(frac * (frames.length - 1))));
}
function tipHtml(i) {
  const frame = frames[i];
  const fb = frame.image_model_feedback;
  const rows = ["<strong>Turn " + frame.turn + "</strong>"];
  if (fb) rows.push("score " + (fb.target_score * 100).toFixed(2) + "%");
  if (hasTraces) {
    const u = frameUsage(frame.turn);
    for (const slot of Object.keys(u.tokens))
      rows.push('<span class="chip" style="display:inline-block;background:' + SEATS[slot] + '"></span> ' +
        esc(names[slot]) + ": " + fmtTokens(u.tokens[slot]) + " out, " +
        ((u.bySlot[slot] || {}).wall_seconds || 0).toFixed(1) + "s");
  }
  return rows.join("<br>");
}
for (const id of ["score-svg", "tokens-svg", "time-svg"]) {
  const svg = document.getElementById(id);
  svg.addEventListener("mousemove", event => {
    const i = chartFrameIndex(svg, event);
    tip.innerHTML = tipHtml(i);
    tip.hidden = false;
    tip.style.left = Math.min(window.innerWidth - 280, event.clientX + 14) + "px";
    tip.style.top = (event.clientY + 14) + "px";
  });
  svg.addEventListener("mouseleave", () => { tip.hidden = true; });
  svg.addEventListener("click", event => go(chartFrameIndex(svg, event)));
}

// Collapsible full trace per agent.
function eventHtml(ev) {
  if (ev.type === "text") return '<p class="ev">' + esc(ev.text) + "</p>";
  if (ev.type === "thinking")
    return '<p class="ev think"><span class="evtag">thinking</span>' + esc(ev.text) + "</p>";
  if (ev.type === "tool_use")
    return '<div class="ev tool">&#8594; <strong>' + esc(ev.name) + "</strong> <code>" + esc(ev.input) + "</code></div>";
  if (ev.type === "tool_result")
    return '<details class="res"><summary>tool result</summary><pre>' + esc(ev.content) + "</pre></details>";
  if (ev.type === "timeout") return '<div class="ev fail">timed out (budget ' + ev.budget_seconds + "s)</div>";
  if (ev.type === "error") return '<div class="ev fail">' + esc(ev.error) + "</div>";
  return "";
}
if (hasTraces) {
  const interviewCards = Object.keys(traces).sort().map(slot => {
    const interview = traces[slot].find(r => r.phase === "interview");
    if (!interview) return "";
    const text = (interview.events || []).filter(e => e.type === "text").map(eventHtml).join("");
    return '<div class="tturn interview"><div class="tthead"><span class="chip" ' +
      'style="display:inline-block;background:' + SEATS[slot] + '"></span> ' +
      esc(names[slot] || ("Agent " + slot)) + "</div>" + (text || '<p class="ev">(no answer)</p>') + "</div>";
  }).join("");
  if (interviewCards) {
    document.getElementById("interviews").style.display = "";
    document.getElementById("interviews-body").innerHTML = interviewCards;
  }
  document.getElementById("traces").innerHTML = Object.keys(traces).sort().map(slot => {
    const recs = turnRecords(traces[slot]);
    const interview = traces[slot].find(r => r.phase === "interview");
    let out = 0, secs = 0, painted = 0;
    for (const r of recs) {
      out += (r.usage && r.usage.output_tokens) || 0;
      secs += r.wall_seconds || 0;
      if (r.painted) painted++;
    }
    const cost = totalCost(traces[slot]);
    const interviewBlock = interview
      ? '<div class="tturn interview"><div class="tthead">post-episode interview</div>' +
        (interview.events || []).map(eventHtml).join("") + "</div>"
      : "";
    const body = recs.map(r =>
      '<div class="tturn"><div class="tthead">turn ' + (r.turn + 1) + " \\u00b7 " + (r.wall_seconds || 0).toFixed(1) +
      "s \\u00b7 " + fmtTokens((r.usage && r.usage.output_tokens) || 0) + " out" +
      (r.painted ? "" : " \\u00b7 NO PAINT") + "</div>" +
      (r.events || []).map(eventHtml).join("") + "</div>"
    ).join("");
    return '<details class="agent"><summary><span class="chip" style="background:' + SEATS[slot] + '"></span>' +
      esc(names[slot] || ("Agent " + slot)) +
      '<span class="meta">painted ' + painted + "/" + recs.length + " turns \\u00b7 " + fmtTokens(out) +
      " tokens out \\u00b7 " + (cost ? "$" + cost.toFixed(2) + " \\u00b7 " : "") +
      (recs.length ? (secs / recs.length).toFixed(1) : 0) + "s avg" +
      (interview ? " \\u00b7 interview" : "") + "</span></summary>" +
      '<div class="trace">' + interviewBlock + body + "</div></details>";
  }).join("");
} else {
  document.getElementById("tokens-card").style.display = "none";
  document.getElementById("time-card").style.display = "none";
  document.getElementById("traces").innerHTML =
    '<p style="color:var(--muted);font-size:12.5px;margin:0">No agent traces for this run \\u2014 ' +
    "only agent seats (run_agent_episode.py) record per-turn traces, tokens, and timing.</p>";
}

let cursor = frames.length - 1;
let timer = null;
function go(f) { cursor = Math.max(0, Math.min(frames.length - 1, f)); show(cursor); }
document.getElementById("prev").addEventListener("click", () => go(cursor - 1));
document.getElementById("next").addEventListener("click", () => go(cursor + 1));
scrub.addEventListener("input", () => go(Number(scrub.value)));
document.getElementById("play").addEventListener("click", function () {
  if (timer) { clearInterval(timer); timer = null; this.textContent = "play"; return; }
  if (cursor >= frames.length - 1) cursor = -1;
  this.textContent = "stop";
  timer = setInterval(() => {
    if (cursor >= frames.length - 1) { clearInterval(timer); timer = null; document.getElementById("play").textContent = "play"; return; }
    go(cursor + 1);
  }, 700);
});
go(cursor);
</script>
"""


def load_run(path: Path) -> tuple[Path, dict, dict, dict]:
    if path.is_dir():
        run_dir = path
        replay = json.loads((run_dir / "replay.json").read_text())
    else:
        run_dir = path.parent
        replay = json.loads(path.read_text())
    results = replay.get("results")
    if results is None:
        results = json.loads((run_dir / "results.json").read_text())
    traces: dict[str, list] = {}
    for trace_file in sorted(run_dir.glob("trace-*.jsonl")):
        slot = trace_file.stem.split("-", 1)[1]
        records = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
        if records:
            traces[slot] = records
    return run_dir, replay, results, traces


def render(input_path: Path, output_path: Path | None = None) -> Path:
    run_dir, replay, results, traces = load_run(input_path)
    if output_path is None:
        output_path = run_dir / "report.html"
    title = "codrawing replay: " + replay["frames"][0]["target"]
    payload = json.dumps(
        {"frames": replay["frames"], "results": results, "traces": traces},
        separators=(",", ":"),
    ).replace("</", "<\\/")
    output_path.write_text(TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload))
    print(f"wrote {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: render_replay_page.py RUN_DIR_OR_REPLAY_JSON [OUTPUT_HTML]")
    render(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) == 3 else None)
