// Replay viewer. The rules live in WebAssembly — this file only feeds the
// module each turn's actions and draws whatever board comes back out, so the
// replay is rendered by the same simulation the server ran.
//
// A replay is JSON: {config:{width,height,maxTurns,teams:[{name,target,region,slots}],
// players:[{name}]}, turns:[{actions:[{slot,x,y,color}], scores:[a,b],
// messages:[{slot,text,public}]}]}.

const SEATS = ["#FB7185", "#60A5FA", "#4ADE80", "#FBBF24",
               "#C084FC", "#F472B6", "#2DD4BF", "#A3E635"];
const TEAM = ["#F5A524", "#38BDF8"];
const OUT_ACCEPTED = 0, OUT_COLLIDED = 1;

const $ = (s) => document.querySelector(s);
const esc = (v) => { const d = document.createElement("div"); d.textContent = v; return d.innerHTML; };

let sim = null, replay = null, cursor = 0, playing = true, timer = null;
let scoreSeries = [[], []], said = {};

async function boot() {
  // Raw WebAssembly: the module exports the rules directly, no glue layer.
  const bytes = await (await fetch("./codrawing.wasm")).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const w = instance.exports;
  sim = {
    mem: () => new Uint32Array(w.memory.buffer),
    mem8: () => new Int8Array(w.memory.buffer),
    boot: w.cwBoot, init: w.cwInit, addTeam: w.cwAddTeam,
    queue: w.cwQueue, step: w.cwStep, setScore: w.cwSetScore,
    canvasPtr: w.cwCanvasPtr, ownersPtr: w.cwOwnersPtr,
    width: w.cwWidth, height: w.cwHeight, turn: w.cwTurn,
    score: w.cwScore, scoreDelta: w.cwScoreDelta, winner: w.cwWinner,
    regionPixels: w.cwRegionPixels, heldByEnemy: w.cwHeldByEnemy,
    accepted: w.cwAccepted, raids: w.cwRaids, outcome: w.cwOutcome,
    teamOf: w.cwTeamOf, regionX: w.cwRegionX, regionW: w.cwRegionW,
  };
  sim.boot();

  const params = new URLSearchParams(location.search);
  const src = params.get("replay") || params.get("replay_uri") || "./replay.json";
  const embedded = document.getElementById("replay-data");
  const raw = embedded ? JSON.parse(embedded.textContent) : await fetchReplay(src);
  replay = raw.frames ? adaptPlatformReplay(raw) : raw;

  buildRoster();
  $("#timeline").max = replay.turns.length;
  $("#max-turns").textContent = replay.config.maxTurns;
  $("#charthint").textContent = replay.config.teams.map((t) => t.target).join(" vs ");
  rewind(0);
  timer = setInterval(() => {
    if (!playing) return;
    if (cursor >= replay.turns.length) { playing = false; $("#play").textContent = "Replay"; return; }
    advance();
  }, 700);
}

async function fetchReplay(src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error("HTTP " + response.status);
  // The Observatory serves replays compressed.
  if (/\.(z|gz)$/.test(new URL(src, location.href).pathname)) {
    const format = src.endsWith(".gz") ? "gzip" : "deflate";
    return new Response(response.body.pipeThrough(new DecompressionStream(format))).json();
  }
  return response.json();
}

// The game server records a replay as {config, frames, results}: a full board
// per turn, not the actions that produced it. The sim replays actions, so the
// writes are recovered from the record.
//
// A pixel written this turn is one whose colour or owner changed, and
// `owners` names the seat that wrote it — the engine stamps an owner on every
// accepted write, erases included. Writes lost to contention changed nothing
// and cannot be located, so they are reported from `collision_slots` alone.
// Replaying the accepted writes reproduces every board exactly.
function adaptPlatformReplay(raw) {
  const cfg = raw.config;
  const size = cfg.width * cfg.height;
  const blank = { canvas: new Array(size).fill("#FFFFFF"), owners: new Array(size).fill(-1) };

  const turns = raw.frames.map((frame, index) => {
    const previous = index ? raw.frames[index - 1] : blank;
    const actions = [];
    for (let p = 0; p < size; p++) {
      if (frame.canvas[p] === previous.canvas[p] && frame.owners[p] === previous.owners[p]) continue;
      const slot = frame.owners[p];
      if (slot < 0) continue;
      actions.push({
        slot,
        x: p % cfg.width,
        y: Math.floor(p / cfg.width),
        color: parseInt(frame.canvas[p].slice(1), 16),
      });
    }
    return {
      actions,
      collided: frame.collision_slots || [],
      scores: (frame.team_feedback || []).map((f) => Math.round((f?.target_score || 0) * 1000)),
      messages: (frame.messages || []).map((m) => ({
        slot: m.slot, text: m.text, public: Boolean(m.public),
      })),
    };
  });

  return {
    config: {
      width: cfg.width,
      height: cfg.height,
      maxTurns: cfg.max_turns,
      players: cfg.players,
      teams: cfg.teams.map((t) => ({
        name: t.name,
        target: t.target,
        region: { x: t.region.x, y: t.region.y, w: t.region.width, h: t.region.height },
        slots: t.slots,
      })),
    },
    turns,
  };
}

// Replaying to turn N means running the sim from turn 0 — the module is the
// only place game state lives, so seeking rebuilds rather than interpolates.
function rewind(target) {
  const cfg = replay.config;
  sim.init(cfg.width, cfg.height, cfg.maxTurns, cfg.players.length);
  for (const t of cfg.teams) sim.addTeam(t.region.x, t.region.y, t.region.w, t.region.h, t.slots[0], t.slots.length);
  scoreSeries = [[], []];
  said = {};
  cursor = 0;
  while (cursor < target) advance(true);
  render();
}

function advance(quiet) {
  const turn = replay.turns[cursor];
  if (!turn) return;
  for (const a of turn.actions || []) sim.queue(a.slot, a.x, a.y, a.color);
  sim.step();
  if (turn.scores) turn.scores.forEach((s, i) => sim.setScore(i, s));
  scoreSeries.forEach((series, i) => series.push(sim.score(i)));
  for (const msg of turn.messages || []) said[msg.slot] = { text: msg.text, turn: cursor + 1, public: msg.public };
  cursor++;
  if (!quiet) render();
}

function buildRoster() {
  const card = (slot) =>
    `<div class="pc" id="pc-${slot}" style="--seat:${SEATS[slot % SEATS.length]}">
       <div class="dot"></div>
       <div><div class="pchead"><span class="pcname">AGENT ${slot}</span>
       <span class="pcstat" id="pcstat-${slot}"></span></div>
       <div class="pcsay" id="pcsay-${slot}">&hellip;</div></div></div>`;
  replay.config.teams.forEach((team, i) => {
    const side = $(i ? "#roster-right" : "#roster-left");
    side.style.setProperty("--team", TEAM[i]);
    side.innerHTML = `<div class="rostername">${esc(team.name)}</div>` + team.slots.map(card).join("");
  });
}

function render() {
  const cfg = replay.config;
  const w = sim.width(), h = sim.height();
  $("#turn").textContent = sim.turn();
  $("#framelabel").textContent = `${cursor} / ${replay.turns.length}`;
  $("#timeline").value = cursor;
  $("#status").textContent = cursor >= replay.turns.length ? "complete" : "replay";

  // Board: read the canvas straight out of the wasm heap.
  const canvas = $("#board"), ctx = canvas.getContext("2d");
  const stage = document.querySelector(".stage");
  const cell = Math.max(4, Math.floor(Math.min((stage.clientWidth - 24) / w, (stage.clientHeight - 24) / h)));
  canvas.width = w * cell; canvas.height = h * cell;
  const pixels = sim.mem().subarray(sim.canvasPtr() >> 2, (sim.canvasPtr() >> 2) + w * h);
  const owners = sim.mem8().subarray(sim.ownersPtr(), sim.ownersPtr() + w * h);

  ctx.fillStyle = "#F7F4EC"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  cfg.teams.forEach((t, i) => {   // each half washed in its team's colour
    ctx.globalAlpha = 0.07; ctx.fillStyle = TEAM[i];
    ctx.fillRect(t.region.x * cell, t.region.y * cell, t.region.w * cell, t.region.h * cell);
    ctx.globalAlpha = 1;
  });
  for (let i = 0; i < w * h; i++) {
    const rgb = pixels[i] & 0xffffff;
    if (rgb === 0xffffff) continue;
    const x = (i % w) * cell, y = Math.floor(i / w) * cell;
    ctx.fillStyle = "#" + rgb.toString(16).padStart(6, "0");
    ctx.fillRect(x + 1, y + 1, cell - 1, cell - 1);
  }
  // Squares held by the other side; an erased one shows as a cross.
  ctx.lineWidth = 1.6;
  for (let i = 0; i < w * h; i++) {
    const owner = owners[i];
    if (owner < 0) continue;
    const x = i % w;
    const region = x < sim.regionX(1) ? 0 : 1;
    if (sim.teamOf(owner) === region) continue;
    const px = x * cell, py = Math.floor(i / w) * cell;
    ctx.strokeStyle = "#E11D48";
    ctx.strokeRect(px + 1.5, py + 1.5, cell - 2, cell - 2);
    if ((pixels[i] & 0xffffff) === 0xffffff) {
      ctx.beginPath();
      ctx.moveTo(px + 4, py + 4); ctx.lineTo(px + cell - 4, py + cell - 4);
      ctx.moveTo(px + cell - 4, py + 4); ctx.lineTo(px + 4, py + cell - 4);
      ctx.stroke();
    }
  }
  const seam = sim.regionX(1) * cell + 0.5;
  ctx.strokeStyle = "rgba(11,15,20,.55)"; ctx.lineWidth = 2; ctx.setLineDash([7, 5]);
  ctx.beginPath(); ctx.moveTo(seam, 0); ctx.lineTo(seam, canvas.height); ctx.stroke();
  ctx.setLineDash([]);

  renderHud();
  renderRoster();
  renderFeed();
  drawSpark();
}

function renderHud() {
  const done = cursor >= replay.turns.length;
  const scores = [sim.score(0), sim.score(1)];
  const best = Math.max(...scores);
  replay.config.teams.forEach((team, k) => {
    const delta = sim.scoreDelta(k), taken = sim.heldByEnemy(k);
    $(k ? "#side-b" : "#side-a").style.setProperty("--team", TEAM[k]);
    $(k ? "#side-b" : "#side-a").innerHTML =
      `<div class="who"><div class="nm">${esc(team.name)}</div><div class="tg">${esc(team.target)}</div></div>
       <div class="score">${(scores[k] / 10).toFixed(0)}<span class="pct">%</span>
         <span class="dl" style="color:${delta >= 0 ? "var(--good)" : "var(--danger)"}">${delta >= 0 ? "+" : ""}${(delta / 10).toFixed(0)}</span></div>
       <div class="tags">
         ${scores[k] === best && best > 0 ? `<span class="chip win">${done ? "winner" : "leading"}</span>` : ""}
         <span class="chip">${sim.regionPixels(k)} px</span>
         ${taken ? `<span class="chip hit">${taken} taken</span>` : ""}
       </div>`;
  });
  const total = scores[0] + scores[1], share = total > 0 ? (scores[0] / total) * 100 : 50;
  $("#tug-a").style.cssText = `width:${share}%;background:${TEAM[0]}`;
  $("#tug-b").style.cssText = `width:${100 - share}%;background:${TEAM[1]}`;

  const banner = $("#banner");
  banner.hidden = !done;
  if (done) {
    const w = sim.winner();
    banner.style.setProperty("--win", w < 0 ? "var(--muted)" : TEAM[w]);
    $("#banner-text").textContent = w < 0 ? "draw" : `${replay.config.teams[w].name} wins`;
  }
}

function renderRoster() {
  replay.config.players.forEach((_, slot) => {
    const stat = $(`#pcstat-${slot}`), say = $(`#pcsay-${slot}`), card = $(`#pc-${slot}`);
    if (!stat) return;
    card.classList.toggle("acted", sim.outcome(slot) === OUT_ACCEPTED);
    const raids = sim.raids(slot);
    stat.innerHTML = `${sim.accepted(slot)} px` + (raids ? ` <span class="atk">${raids} raid</span>` : "");
    const entry = said[slot];
    const fresh = entry && entry.turn === sim.turn();
    say.classList.toggle("old", Boolean(entry) && !fresh);
    say.innerHTML = entry
      ? (fresh ? "" : `<span class="stamp">T${entry.turn}</span> `) + esc(entry.text)
      : "has not spoken yet";
  });
}

function renderFeed() {
  const turn = replay.turns[cursor - 1];
  const rows = [];
  for (const a of (turn && turn.actions) || []) {
    const accepted = sim.outcome(a.slot) === OUT_ACCEPTED;
    const region = a.x < sim.regionX(1) ? 0 : 1;
    const raid = accepted && sim.teamOf(a.slot) !== region;
    const seat = `<span style="color:${SEATS[a.slot % SEATS.length]}">AGENT ${a.slot}</span>`;
    const at = `(${a.x},${a.y})`;
    rows.push(
      `<div class="ev${raid ? " raid" : ""}"><span class="t">T${sim.turn()}</span>` +
      (raid ? `<span class="tag">RAID</span>` : "") +
      `<span>${seat} ${accepted ? (raid ? "wiped " + at + " in " + esc(replay.config.teams[region].name) : "painted " + at) : "collided at " + at}</span></div>`);
  }
  // A replay recovered from recorded boards knows which seats collided but not
  // where, since a lost write leaves no trace on the canvas.
  for (const slot of (turn && turn.collided) || []) {
    const seat = `<span style="color:${SEATS[slot % SEATS.length]}">AGENT ${slot}</span>`;
    rows.push(`<div class="ev"><span class="t">T${sim.turn()}</span>` +
      `<span>${seat} collided</span></div>`);
  }
  const feed = $("#feed");
  feed.innerHTML = rows.join("") || `<div class="ev"><span class="t">T${sim.turn()}</span><span style="color:var(--muted)">no accepted writes</span></div>`;
  feed.scrollTop = 0;
}

function drawSpark() {
  const c = $("#spark"), ctx = c.getContext("2d");
  const w = c.clientWidth || 300, h = Math.max(40, c.clientHeight || 40);
  c.width = w; c.height = h;
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 2, r: 40, t: 8, b: 4 };
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const len = Math.max(scoreSeries[0].length, 1);
  const max = Math.max(50, ...scoreSeries.flat());
  const x = (i) => pad.l + (len > 1 ? (i / (len - 1)) * plotW : plotW / 2);
  const y = (v) => pad.t + plotH - (v / max) * plotH;
  scoreSeries.forEach((series, k) => {
    if (!series.length) return;
    ctx.beginPath(); ctx.moveTo(x(0), pad.t + plotH);
    series.forEach((v, i) => ctx.lineTo(x(i), y(v)));
    ctx.lineTo(x(series.length - 1), pad.t + plotH); ctx.closePath();
    ctx.globalAlpha = .16; ctx.fillStyle = TEAM[k]; ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = TEAM[k]; ctx.lineWidth = 2;
    ctx.beginPath(); series.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)))); ctx.stroke();
    ctx.fillStyle = TEAM[k]; ctx.font = "700 11px ui-monospace,monospace";
    ctx.fillText((series[series.length - 1] / 10).toFixed(0) + "%", w - pad.r + 5, y(series[series.length - 1]) + 4);
  });
}

$("#play").addEventListener("click", function () {
  if (!playing && cursor >= replay.turns.length) rewind(0);
  playing = !playing;
  this.textContent = playing ? "Pause" : "Play";
});
$("#timeline").addEventListener("input", (e) => {
  playing = false; $("#play").textContent = "Play";
  rewind(Number(e.target.value));
});
window.addEventListener("resize", () => sim && render());

boot();
