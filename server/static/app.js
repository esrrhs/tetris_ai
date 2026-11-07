// Tetris AI web client.
// The backend owns the game and the policy: every POST /api/game/step returns
// one placement decision plus the exact frame sequence of that piece. This
// file only replays those frames on the canvas and renders the state DTO.
(() => {
  "use strict";

  const CELL = 32;
  const W = 10 * CELL;
  const H = 20 * CELL;
  // Cell value = pieceType + 1 (I=1 ... Z=7), matching tetris::Board::Lock.
  const COLORS = [null, "#38d9f5", "#ffa63d", "#557dff", "#c07bff",
                  "#ffd84d", "#4ee08a", "#ff6482"];
  // Rotation-0 shapes, same tables as src/tetris/shapes.cpp.
  const SHAPES = {
    I: [[-1, 0], [0, 0], [1, 0], [2, 0]],
    L: [[-1, 0], [0, 0], [1, 0], [-1, 1]],
    J: [[-1, 0], [0, 0], [1, 0], [1, 1]],
    T: [[-1, 0], [0, 0], [1, 0], [0, 1]],
    O: [[0, 0], [1, 0], [0, 1], [1, 1]],
    S: [[0, 0], [1, 0], [-1, 1], [0, 1]],
    Z: [[-1, 0], [0, 0], [0, 1], [1, 1]],
  };
  const KEY_OPS = {
    ArrowLeft: ["Left"], ArrowRight: ["Right"], ArrowUp: ["RotateCW"],
    ArrowDown: [], " ": ["HardDrop"], z: ["RotateCCW"], Z: ["RotateCCW"],
  };

  let sessionId = localStorage.getItem("tetris_ai_session");
  if (!sessionId) {
    sessionId = "sess_" + Date.now() + "_" + Math.random().toString(36).slice(2, 9);
    localStorage.setItem("tetris_ai_session", sessionId);
  }

  let current = null;
  let auto = false;
  // Opening the page starts the policy immediately; `?auto=0` opts out for
  // stepping / manual play. Pausing the button turns this off too.
  let autoByDefault = new URLSearchParams(window.location.search).get("auto") !== "0";
  let busy = false;
  let manual = false;
  let epoch = 0;
  let toastTimer = null;
  const logs = [];

  const $ = (id) => document.getElementById(id);
  const canvas = $("board");
  const ctx = canvas.getContext("2d");

  function setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, Math.max(0, ms)));
  const speed = () => Number($("speed").value);
  const frameDelay = () => Math.round(86 - speed() * 0.76);  // 86ms .. 10ms

  function tint(hex, amount) {
    const n = parseInt(hex.slice(1), 16);
    const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
    const mix = (v) => Math.round(v + (255 - v) * amount);
    return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
  }
  function rgba(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function block(x, y, colorValue, opts = {}) {
    if (y < 0 || y >= 20 || x < 0 || x >= 10) return;
    const idx = (colorValue > 0 && colorValue < COLORS.length) ? colorValue : 1;
    const color = COLORS[idx];
    const px = x * CELL, py = y * CELL;
    if (opts.ghost) {
      ctx.save();
      ctx.strokeStyle = rgba(color, 0.55);
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      roundRect(px + 3, py + 3, CELL - 6, CELL - 6, 4);
      ctx.stroke();
      ctx.restore();
      return;
    }
    ctx.save();
    ctx.globalAlpha = opts.alpha === undefined ? 1 : opts.alpha;
    ctx.shadowColor = rgba(color, 0.65);
    ctx.shadowBlur = opts.flat ? 0 : 9;
    const grad = ctx.createLinearGradient(px, py, px + CELL, py + CELL);
    grad.addColorStop(0, tint(color, 0.35));
    grad.addColorStop(1, color);
    ctx.fillStyle = grad;
    roundRect(px + 1.5, py + 1.5, CELL - 3, CELL - 3, 5);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "rgba(255,255,255,.30)";
    roundRect(px + 4, py + 4, CELL - 8, 5, 2); ctx.fill();
    ctx.fillStyle = "rgba(0,0,0,.18)";
    roundRect(px + 4, py + CELL - 8, CELL - 8, 4, 2); ctx.fill();
    ctx.restore();
  }

  function drawBase(board, flashRows = []) {
    ctx.fillStyle = "#091120";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(140,170,220,.055)";
    ctx.lineWidth = 1;
    for (let x = 1; x < 10; ++x) {
      ctx.beginPath(); ctx.moveTo(x * CELL + .5, 0); ctx.lineTo(x * CELL + .5, H); ctx.stroke();
    }
    for (let y = 1; y < 20; ++y) {
      ctx.beginPath(); ctx.moveTo(0, y * CELL + .5); ctx.lineTo(W, y * CELL + .5); ctx.stroke();
    }
    if (board) {
      for (let y = 0; y < 20; ++y) {
        for (let x = 0; x < 10; ++x) {
          const v = board[y] ? board[y][x] : 0;
          if (v) block(x, y, v, { flat: true });
        }
      }
    }
    if (flashRows && flashRows.length) {
      for (const r of flashRows) {
        ctx.fillStyle = "rgba(255,255,255,.78)";
        ctx.fillRect(0, r * CELL + 1, W, CELL - 2);
      }
    }
  }

  function drawCells(cells, colorValue, opts = {}) {
    if (!cells) return;
    for (const c of cells) block(c[0], c[1], colorValue, opts);
  }

  function renderBoard(state) {
    drawBase(state.board);
    if (state.active && state.active.cells) {
      const color = state.active.typeId + 1;
      if (state.ghost && state.ghost.cells) {
        drawCells(state.ghost.cells, color, { ghost: true });
      }
      drawCells(state.active.cells, color);
    }
  }

  async function animatePiece(a, state, myEpoch) {
    const frames = a.frames || [];
    const color = a.typeId + 1;
    const delay = frameDelay();
    for (let i = 0; i < frames.length; ++i) {
      if (myEpoch !== epoch) return;
      drawBase(a.boardBefore);
      drawCells(a.lockedCells, color, { ghost: true });
      drawCells(frames[i].cells, color);
      await sleep(i === frames.length - 1 ? delay * 1.5 : delay);
    }
    if (myEpoch !== epoch) return;
    if (a.clearedRows && a.clearedRows.length) {
      drawBase(a.boardBefore, a.clearedRows);
      drawCells(a.lockedCells, color);
      await sleep(Math.max(90, delay * 2.4));
      if (myEpoch !== epoch) return;
    }
    renderBoard(state);
  }

  // ---------- networking ----------
  async function api(path, options = {}) {
    const headers = Object.assign({ "X-Session-Id": sessionId }, options.headers || {});
    if (options.body) headers["Content-Type"] = "application/json";
    const res = await fetch(path, Object.assign({}, options, { headers }));
    const text = await res.text();
    let dto;
    try { dto = JSON.parse(text); } catch (e) { throw new Error("响应不是 JSON: " + text.slice(0, 80)); }
    if (res.headers.get("X-Session-Id")) sessionId = res.headers.get("X-Session-Id");
    return dto;
  }

  function toast(message) {
    const el = $("toast");
    el.textContent = message;
    el.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), 2600);
  }

  function setBusy(v) {
    busy = v;
    const dot = $("liveDot");
    dot.className = "live-dot" + (v ? " busy" : " on");
    $("btnStep").disabled = v || manual;
    $("btnBurst").disabled = v || manual;
    $("btnNew").disabled = v;
    $("btnNewTop").disabled = v;
  }

  function speedLabel() {
    const s = speed();
    $("speedLabel").textContent = s < 25 ? "慢" : s < 70 ? "中" : "快";
  }

  // ---------- state rendering ----------
  function miniPiece(type) {
    const cells = SHAPES[type] || SHAPES.O;
    const xs = cells.map((c) => c[0]), ys = cells.map((c) => c[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys);
    const offX = Math.floor((4 - (maxX - minX + 1)) / 2);
    const grid = document.createElement("div");
    grid.className = "mini-grid";
    const filled = new Set(cells.map((c) => `${c[0] - minX + offX},${c[1] - minY}`));
    for (let y = 0; y < 2; ++y) {
      for (let x = 0; x < 4; ++x) {
        const cell = document.createElement("div");
        cell.className = "mini-cell" + (filled.has(`${x},${y}`) ? " filled" : "");
        if (filled.has(`${x},${y}`)) {
          const idx = ["I", "L", "J", "T", "O", "S", "Z"].indexOf(type) + 1;
          cell.style.background = COLORS[idx];
          cell.style.color = COLORS[idx];
        }
        grid.appendChild(cell);
      }
    }
    return grid;
  }

  function renderNext(next) {
    const box = $("nextList");
    box.innerHTML = "";
    (next || []).forEach((type, i) => {
      const item = document.createElement("div");
      item.className = "next-item";
      const name = document.createElement("div");
      name.className = "next-name";
      // `next` is the five pieces AFTER the active one (Bag7::PeekNext).
      name.textContent = (i + 1) + "·" + type;
      item.appendChild(miniPiece(type));
      item.appendChild(name);
      box.appendChild(item);
    });
  }

  function renderDecision(d) {
    const src = $("decisionSource"), main = $("decisionMain"), box = $("candidates");
    box.innerHTML = "";
    if (!d) {
      src.textContent = "—";
      main.textContent = "等待第一次推进";
      return;
    }
    const isModel = d.source === "model";
    src.textContent = isModel ? "神经网络" : (d.source === "heuristic" ? "启发式教师" : d.source);
    main.innerHTML = `<b>${src.textContent}</b> · ${Number(d.timeMs).toFixed(1)} ms · ` +
      `落点 <em>r${d.rot} x${d.x} y${d.landingY}</em>` +
      (d.linesCleared ? ` · 消 <em>${d.linesCleared}</em> 行` : "") +
      (d.scoreGain ? ` · +${d.scoreGain} 分` : "");

    const cs = d.candidates || [];
    if (!cs.length) return;
    const vals = cs.map((c) => isModel ? c.probability : c.score);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    cs.forEach((c, i) => {
      const v = vals[i];
      const pct = hi > lo ? 12 + 88 * (v - lo) / (hi - lo) : 85;
      const row = document.createElement("div");
      row.className = "candidate";
      const label = document.createElement("div");
      label.className = "cand-label";
      label.textContent = `r${c.rot} x${c.x} y${c.landingY}`;
      const track = document.createElement("div");
      track.className = "cand-track";
      const fill = document.createElement("div");
      fill.className = "cand-fill";
      fill.style.width = Math.max(4, Math.min(100, pct)) + "%";
      track.appendChild(fill);
      const value = document.createElement("div");
      value.className = "cand-value";
      value.textContent = isModel ? v.toFixed(1) + "%" : v.toFixed(0);
      row.append(label, track, value);
      box.appendChild(row);
    });
  }

  function addLog(state, burst) {
    const d = state.decision;
    let text;
    if (burst) {
      text = `快进 ${burst} 步 → STEP ${state.step} · ${state.lines} 行 · ${state.score} 分`;
    } else if (d) {
      const a = state.animation || {};
      const name = a.type || (state.active ? state.active.type : "?");
      text = `STEP ${state.step} · ${name} · r${d.rot} x${d.x} y${d.landingY}` +
        (d.linesCleared ? ` · 消 ${d.linesCleared} 行` : "") +
        (d.scoreGain ? ` · +${d.scoreGain}` : "");
    } else {
      text = `STEP ${state.step} · 手动推进`;
    }
    logs.unshift({ text, clear: !!(d && d.linesCleared) });
    if (logs.length > 10) logs.length = 10;
    const box = $("eventLog");
    box.innerHTML = "";
    for (const entry of logs) {
      const line = document.createElement("div");
      line.className = entry.clear ? "clear" : "";
      const b = document.createElement("b");
      b.textContent = entry.text.split(" · ")[0];
      line.appendChild(b);
      line.appendChild(document.createTextNode(entry.text.slice(b.textContent.length)));
      box.appendChild(line);
    }
  }

  function renderState(state) {
    if (!state || state.error) {
      if (state && state.error) toast(state.error);
      return;
    }
    current = state;
    renderBoard(state);
    $("statScore").textContent = state.score.toLocaleString();
    $("statLines").textContent = state.lines;
    $("statPieces").textContent = state.pieces;
    $("statCombo").textContent = state.combo;
    $("stepLabel").textContent = "STEP " + state.step;
    const m = state.metrics || {};
    $("metricHoles").textContent = (m.holes || 0).toFixed(0);
    $("metricBump").textContent = (m.bumpiness || 0).toFixed(0);
    $("metricHeight").textContent = (m.maxHeight || 0).toFixed(0);
    renderNext(state.next);
    renderDecision(state.decision);
    $("sessionLabel").textContent = (state.sessionId || "").slice(0, 18);
    $("statusLine").textContent = `STEP ${state.step} · ` +
      `${state.backend === "model" ? "神经网络" : "启发式教师"} · seed ${state.seed}`;
    const overlay = $("boardOverlay");
    if (state.gameOver) {
      $("overlayTitle").textContent = "游戏结束";
      $("overlaySub").textContent = `存活 ${state.pieces} 块 · 消 ${state.lines} 行 · 得分 ${state.score}\n点击「再来一局」重新发牌`;
      overlay.classList.remove("hidden");
      stopAuto();
    } else {
      overlay.classList.add("hidden");
    }
    $("liveDot").className = "live-dot on";
  }

  async function refreshStatus() {
    try {
      const s = await api("/api/status");
      const ok = s.status === "ok";
      $("backendBadge").textContent = ok ? "服务在线 :" + s.port : "服务异常";
      $("backendBadge").className = "badge" + (ok ? "" : " warn");
      $("modelBadge").textContent = s.modelAvailable
        ? "模型 · " + String(s.backend || "").replace("model:", "")
        : "启发式教师（无模型）";
      $("modelBadge").className = "badge badge-ghost" + (s.modelAvailable ? "" : " warn");
      if (!s.modelAvailable) {
        $("mode").value = "heuristic";
        $("mode").disabled = true;
      } else {
        $("mode").disabled = false;
      }
    } catch (e) {
      $("backendBadge").textContent = "服务未连接";
      $("backendBadge").className = "badge warn";
      $("statusLine").textContent = "无法连接后端：" + e.message;
    }
  }

  // ---------- driving ----------
  async function stepOnce(count = 1) {
    if (busy) return null;
    setBusy(true);
    const myEpoch = epoch;
    try {
      const dto = await api("/api/game/step", {
        method: "POST",
        body: JSON.stringify({ count }),
      });
      if (dto.error) { toast(dto.error); stopAuto(); return dto; }
      if (count === 1 && dto.animation) {
        await animatePiece(dto.animation, dto, myEpoch);
      }
      if (myEpoch !== epoch) return dto;
      renderState(dto);
      addLog(dto, count > 1 ? count : 0);
      return dto;
    } catch (e) {
      toast("请求失败：" + e.message);
      stopAuto();
      return null;
    } finally {
      if (myEpoch === epoch) setBusy(false);
    }
  }

  async function tick(ops) {
    if (busy || !manual) return;
    setBusy(true);
    try {
      const dto = await api("/api/game/tick", {
        method: "POST", body: JSON.stringify({ ops }),
      });
      if (dto.error) { toast(dto.error); return; }
      renderState(dto);
      addLog(dto, 0);
    } catch (e) {
      toast("请求失败：" + e.message);
    } finally {
      setBusy(false);
    }
  }

  async function autoLoop() {
    while (auto && !manual) {
      const dto = await stepOnce(1);
      if (!dto || dto.error || dto.gameOver) { stopAuto(); break; }
      await sleep(Math.max(0, frameDelay() * 1.3));
    }
  }

  function startAuto() {
    if (manual || auto) return;
    auto = true;
    $("btnAuto").textContent = "暂停播放";
    $("btnAuto").classList.add("running");
    autoLoop();
  }
  function stopAuto() {
    auto = false;
    $("btnAuto").textContent = "自动播放";
    $("btnAuto").classList.remove("running");
  }

  async function newGame() {
    stopAuto();
    ++epoch;
    setBusy(true);
    let ok = false;
    const seed = Number($("seed").value) || 0;
    const mode = $("mode").value;
    try {
      const dto = await api("/api/game/new", {
        method: "POST", body: JSON.stringify({ seed, mode }),
      });
      if (dto.error) { toast(dto.error); return; }
      logs.length = 0;
      $("eventLog").innerHTML = "";
      $("decisionMain").textContent = "等待第一次推进";
      $("candidates").innerHTML = "";
      renderState(dto);
      addLog(dto, 0);
      $("statusLine").textContent = `新局已开始 · ${dto.backend === "model" ? "神经网络" : "启发式教师"} · seed ${dto.seed}`;
      ok = true;
    } catch (e) {
      toast("新局失败：" + e.message);
    } finally {
      setBusy(false);
      // Keep watching after "play again" / mode change without another click.
      if (ok && autoByDefault && !manual) startAuto();
    }
  }

  function setManual(v) {
    manual = v;
    if (v) stopAuto();
    $("manualHelp").classList.toggle("hidden", !v);
    $("btnAuto").disabled = v;
    $("btnStep").disabled = v;
    $("btnBurst").disabled = v;
    $("statusLine").textContent = v
      ? "手动模式：← → 平移 · ↑ 旋转 · ↓ 下移 · 空格 直落 · Z 反转"
      : (autoByDefault ? "自动运行中 · 点「暂停播放」可暂停"
                       : (current ? `STEP ${current.step} · 按「单步推进」或「自动播放」驱动` : "已连接"));
    if (!v && autoByDefault) startAuto();
  }

  // ---------- bindings ----------
  function bind() {
    $("btnStep").addEventListener("click", () => stepOnce(1));
    $("btnBurst").addEventListener("click", () => stepOnce(10));
    $("btnAuto").addEventListener("click", () => {
      if (auto) { autoByDefault = false; stopAuto(); }
      else { autoByDefault = true; startAuto(); }
    });
    $("btnNew").addEventListener("click", newGame);
    $("btnNewTop").addEventListener("click", newGame);
    $("btnOverlayRestart").addEventListener("click", newGame);
    $("btnRules").addEventListener("click", () => $("rulesModal").classList.remove("hidden"));
    $("btnRulesClose").addEventListener("click", () => $("rulesModal").classList.add("hidden"));
    $("rulesModal").addEventListener("click", (e) => {
      if (e.target === $("rulesModal")) $("rulesModal").classList.add("hidden");
    });
    $("speed").addEventListener("input", speedLabel);
    $("mode").addEventListener("change", newGame);
    $("manualToggle").addEventListener("change", (e) => setManual(e.target.checked));
    document.addEventListener("keydown", (e) => {
      if (!manual) return;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
      if (!(e.key in KEY_OPS)) return;
      e.preventDefault();
      tick(KEY_OPS[e.key]);
    });
  }

  document.addEventListener("DOMContentLoaded", async () => {
    setupCanvas();
    drawBase(null);
    bind();
    speedLabel();
    // Browsers may restore a previously checked manual-mode box on reload.
    if ($("manualToggle").checked) setManual(true);
    await refreshStatus();
    try {
      const dto = await api("/api/game/state");
      renderState(dto);
    } catch (e) {
      $("statusLine").textContent = "状态获取失败：" + e.message;
    }
    // Autoplay is the default; ?auto=0 (or manual mode) opts out. A stored
    // session that already ended gets a fresh game instead of a dead board.
    if (autoByDefault && !manual) {
      if (current && current.gameOver) newGame();
      else startAuto();
    }
  });
})();
