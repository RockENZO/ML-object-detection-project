(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const base = (params.get('data') || 'data/').replace(/\/?$/, '/');
  const KIND_TEAM = ['A', 'B'];
  const GLYPH = {
    momentum_swing: '⇄',
    sustained_pressure: '▲',
    possession_change: '◐',
    box_entry: '▣',
    fast_ball: '➤',
    tempo_surge: '⚡',
    crowding: '◎',
  };
  const NEUTRAL = '#c9d4e3';

  let D, frames, video, W, H, L, PW, names, colours;
  let last = { t: -1, w: 0, h: 0, key: '' };
  let heat = null;
  let notesKey = '';
  let selected = null;
  let sortKey = 'distance_m';
  let sortDir = -1;
  let playerList = [];
  let playerById = new Map();
  let lastSt = null;

  const fmt = (t) => {
    t = Math.max(0, t);
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
  };
  const fmt1 = (t) => `${fmt(t)}.${Math.floor((t % 1) * 10)}`;
  const esc = (value) =>
    String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, k) => a + (b - a) * k;
  const teamColour = (team) => (team ? colours[team] : NEUTRAL);
  const hexA = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  };

  function sizeCanvas(canvas, cssW, cssH) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(cssW * dpr));
    const h = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  /* ---------- data access ---------- */

  function indexAt(t) {
    let lo = 0;
    let hi = frames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (frames[mid].t <= t) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  function rowMap(frame) {
    if (!frame._m) frame._m = new Map(frame.p.map((r) => [r[0], r]));
    return frame._m;
  }

  /** Interpolated players and ball at video time t (no interpolation across cuts). */
  function stateAt(t) {
    const i = indexAt(t);
    const f = frames[i];
    const n = frames[i + 1];
    let k = 0;
    if (n && n.s === f.s && n.e && f.e && n.t - f.t <= 0.25 && t > f.t) k = (t - f.t) / (n.t - f.t);
    const players = [];
    const next = k > 0 ? rowMap(n) : null;
    for (const r of f.p) {
      const m = next ? next.get(r[0]) : null;
      const p = { id: r[0], kind: r[1], x: r[2], y: r[3], o: r[4], box: r.length >= 9 ? r.slice(5, 9) : null };
      if (m) {
        p.x = lerp(r[2], m[2], k);
        p.y = lerp(r[3], m[3], k);
        if (p.box && m.length >= 9) p.box = p.box.map((v, j) => lerp(v, m[5 + j], k));
      }
      players.push(p);
    }
    let ball = null;
    if (f.b) {
      ball = { x: f.b[0], y: f.b[1], o: f.b[2], px: f.b[3], py: f.b[4], air: f.b[5] };
      if (n && k > 0 && n.b) {
        ball.x = lerp(f.b[0], n.b[0], k);
        ball.y = lerp(f.b[1], n.b[1], k);
        if (f.b[3] != null && n.b[3] != null) {
          ball.px = lerp(f.b[3], n.b[3], k);
          ball.py = lerp(f.b[4], n.b[4], k);
        }
      }
    }
    return { i, f, players, ball };
  }

  function momentumAt(t) {
    const b = D.bin_seconds;
    const i = clamp(Math.floor((t - D.momentum[0].t) / b), 0, D.momentum.length - 1);
    return D.momentum[i];
  }

  const trackNo = (id) => id % 10000;
  const shotOf = (id) => Math.floor(id / 10000);
  const tag = (id) => `#${trackNo(id)}${shotOf(id) ? `·s${shotOf(id)}` : ''}`;
  const spaceAt = (t) => {
    const series = D.space_control && D.space_control.series;
    if (!series || !series.length) return null;
    const row = series[clamp(Math.floor(t), 0, series.length - 1)];
    return row && row.A != null ? row.A : null;
  };

  /** Velocity (m/s) from observed positions a few samples either side; null when it cannot be trusted. */
  function velocity(i, id) {
    const f = frames[i];
    const pick = (dir) => {
      for (let k = 3; k >= 1; k--) {
        const g = frames[i + dir * k];
        if (!g || g.s !== f.s || !g.e) continue;
        const r = rowMap(g).get(id);
        if (r && r[4]) return { t: g.t, x: r[2], y: r[3] };
      }
      return null;
    };
    const a = pick(-1);
    const b = pick(1);
    if (!a || !b || b.t <= a.t) return null;
    const vx = (b.x - a.x) / (b.t - a.t);
    const vy = (b.y - a.y) / (b.t - a.t);
    const speed = Math.hypot(vx, vy);
    return speed > 12 ? null : { vx, vy, speed };
  }

  /* ---------- polygon helpers for the camera footprint ---------- */

  function clipLinear(poly, a, b, c) {
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i];
      const q = poly[(i + 1) % poly.length];
      const fp = a * p[0] + b * p[1] + c;
      const fq = a * q[0] + b * q[1] + c;
      if (fp >= 0) out.push(p);
      if ((fp >= 0) !== (fq >= 0)) {
        const k = fp / (fp - fq);
        out.push([lerp(p[0], q[0], k), lerp(p[1], q[1], k)]);
      }
    }
    return out;
  }

  function footprint(h) {
    if (!h) return null;
    let poly = [[0, 0], [W, 0], [W, H], [0, H]];
    poly = clipLinear(poly, h[6], h[7], h[8] - 1e-4);
    if (poly.length < 3) return null;
    poly = poly.map(([x, y]) => {
      const w = h[6] * x + h[7] * y + h[8];
      return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
    });
    poly = clipLinear(poly, 1, 0, 0);
    poly = clipLinear(poly, -1, 0, L);
    poly = clipLinear(poly, 0, 1, 0);
    poly = clipLinear(poly, 0, -1, PW);
    return poly.length >= 3 ? poly : null;
  }

  /* ---------- pitch ---------- */

  function pitchGeometry(canvas) {
    const pad = 3;
    const cssW = canvas.parentElement.clientWidth - 28;
    const scale = cssW / (L + 2 * pad);
    const cssH = (PW + 2 * pad) * scale;
    return { pad, scale, cssW, cssH };
  }

  function drawPitchMarkings(ctx, g) {
    const X = (x) => (g.pad + x) * g.scale;
    const Y = (y) => (g.pad + y) * g.scale;
    ctx.fillStyle = '#134d31';
    ctx.fillRect(0, 0, g.cssW, g.cssH);
    const stripes = 14;
    for (let i = 0; i < stripes; i++) {
      ctx.fillStyle = i % 2 ? 'rgba(255,255,255,0.028)' : 'rgba(0,0,0,0.05)';
      ctx.fillRect(X((L / stripes) * i), Y(0), (L / stripes) * g.scale, PW * g.scale);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.82)';
    ctx.lineWidth = Math.max(1, g.scale * 0.14);
    ctx.strokeRect(X(0), Y(0), L * g.scale, PW * g.scale);
    const line = (x1, y1, x2, y2) => {
      ctx.beginPath();
      ctx.moveTo(X(x1), Y(y1));
      ctx.lineTo(X(x2), Y(y2));
      ctx.stroke();
    };
    const circle = (x, y, r, a0 = 0, a1 = Math.PI * 2) => {
      ctx.beginPath();
      ctx.arc(X(x), Y(y), r * g.scale, a0, a1);
      ctx.stroke();
    };
    const dot = (x, y) => {
      ctx.beginPath();
      ctx.arc(X(x), Y(y), Math.max(1.5, g.scale * 0.22), 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.82)';
      ctx.fill();
    };
    line(L / 2, 0, L / 2, PW);
    circle(L / 2, PW / 2, 9.15);
    dot(L / 2, PW / 2);
    for (const side of [0, 1]) {
      const s = side ? -1 : 1;
      const x0 = side ? L : 0;
      const box = (depth, width) =>
        ctx.strokeRect(
          X(side ? L - depth : 0),
          Y((PW - width) / 2),
          depth * g.scale,
          width * g.scale,
        );
      box(16.5, 40.32);
      box(5.5, 18.32);
      dot(x0 + s * 11, PW / 2);
      // Penalty arc: the part of the 9.15 m circle outside the box.
      const a = Math.acos(5.5 / 9.15);
      circle(x0 + s * 11, PW / 2, 9.15, side ? Math.PI - a : -a, side ? Math.PI + a : a);
      // Goal frame.
      ctx.save();
      ctx.lineWidth = Math.max(2, g.scale * 0.3);
      line(x0, PW / 2 - 3.66, x0, PW / 2 + 3.66);
      ctx.restore();
    }
    for (const [cx, cy, a0] of [[0, 0, 0], [L, 0, Math.PI / 2], [L, PW, Math.PI], [0, PW, -Math.PI / 2]]) {
      circle(cx, cy, 1, a0, a0 + Math.PI / 2);
    }
  }

  function drawHeat(ctx, g) {
    const grid = heat.grid;
    const cw = L / heat.cols;
    const ch = PW / heat.rows;
    for (let r = 0; r < heat.rows; r++) {
      for (let c = 0; c < heat.cols; c++) {
        const v = grid[r * heat.cols + c] / heat.max;
        if (v <= 0.02) continue;
        ctx.fillStyle = hexA(colours[heat.team], Math.pow(v, 0.65) * 0.6);
        ctx.fillRect((g.pad + c * cw) * g.scale, (g.pad + r * ch) * g.scale, cw * g.scale + 0.5, ch * g.scale + 0.5);
      }
    }
  }

  function computeHeat(team) {
    if (!team) {
      heat = null;
      return;
    }
    const cols = 30;
    const rows = 20;
    const grid = new Float32Array(cols * rows);
    const kind = team === 'A' ? 0 : 1;
    for (const f of frames) {
      if (!f.e) continue;
      for (const r of f.p) {
        if (r[1] !== kind || !r[4]) continue;
        const c = clamp(Math.floor((r[2] / L) * cols), 0, cols - 1);
        const row = clamp(Math.floor((r[3] / PW) * rows), 0, rows - 1);
        grid[row * cols + c] += 1;
      }
    }
    heat = { team, cols, rows, grid, max: Math.max(1, ...grid) };
  }

  function drawSpace(ctx, g, st) {
    const a = [];
    const b = [];
    for (const p of st.players) {
      if (!p.o) continue;
      if (p.kind === 0) a.push(p);
      else if (p.kind === 1) b.push(p);
    }
    if (a.length < 3 || b.length < 3) return;
    const poly = footprint(st.f.h);
    if (!poly) return;
    ctx.save();
    ctx.beginPath();
    poly.forEach(([x, y], i) => (i ? ctx.lineTo((g.pad + x) * g.scale, (g.pad + y) * g.scale) : ctx.moveTo((g.pad + x) * g.scale, (g.pad + y) * g.scale)));
    ctx.closePath();
    ctx.clip();
    const step = 1;
    const px = (v) => Math.round((g.pad + v) * g.scale);
    const near = (list, x, y) => {
      let best = Infinity;
      for (const p of list) best = Math.min(best, (p.x - x) ** 2 + (p.y - y) ** 2);
      return best;
    };
    const fillA = hexA(colours.A, 0.3);
    const fillB = hexA(colours.B, 0.3);
    for (let y = 0; y < PW; y += step) {
      for (let x = 0; x < L; x += step) {
        ctx.fillStyle = near(a, x + step / 2, y + step / 2) < near(b, x + step / 2, y + step / 2) ? fillA : fillB;
        // Shared integer edges: no seams or double-painted overlaps between neighbouring cells.
        ctx.fillRect(px(x), px(y), px(x + step) - px(x), px(y + step) - px(y));
      }
    }
    ctx.restore();
  }

  function arrow(ctx, x1, y1, x2, y2, colour, width) {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const head = Math.max(5, width * 3.2);
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(angle - 0.45), y2 - head * Math.sin(angle - 0.45));
    ctx.lineTo(x2 - head * Math.cos(angle + 0.45), y2 - head * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fill();
  }

  function drawSelectedPath(ctx, g, st) {
    const X = (x) => (g.pad + x) * g.scale;
    const Y = (y) => (g.pad + y) * g.scale;
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(255,210,63,0.75)';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    let open = false;
    let prev = null;
    for (const f of frames) {
      const r = f.e ? rowMap(f).get(selected) : null;
      if (!r || !r[4] || f.s !== st.f.s) {
        open = false;
        prev = null;
        continue;
      }
      if (prev && Math.hypot(r[2] - prev[0], r[3] - prev[1]) > 6) open = false;
      if (open) ctx.lineTo(X(r[2]), Y(r[3]));
      else ctx.moveTo(X(r[2]), Y(r[3]));
      open = true;
      prev = [r[2], r[3]];
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawPitch(t, st) {
    const canvas = $('pitch');
    const g = pitchGeometry(canvas);
    canvas.style.height = `${g.cssH}px`;
    const ctx = sizeCanvas(canvas, g.cssW, g.cssH);
    const X = (x) => (g.pad + x) * g.scale;
    const Y = (y) => (g.pad + y) * g.scale;
    drawPitchMarkings(ctx, g);
    if (heat) drawHeat(ctx, g);
    if ($('tSpace').checked && st.f.e) drawSpace(ctx, g, st);

    // Penalty-area highlight for an active box moment.
    for (const m of D.moments) {
      if (m.type !== 'box_entry' || t < m.start || t > m.end) continue;
      const right = /right/.test(m.title);
      ctx.fillStyle = hexA(teamColour(m.team), 0.18);
      ctx.fillRect(X(right ? L - 16.5 : 0), Y((PW - 40.32) / 2), 16.5 * g.scale, 40.32 * g.scale);
    }

    if ($('tFootprint').checked && st.f.e) {
      const poly = footprint(st.f.h);
      if (poly) {
        ctx.beginPath();
        poly.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
        ctx.closePath();
        ctx.fillStyle = 'rgba(255,255,255,0.09)';
        ctx.fill();
        ctx.setLineDash([6, 5]);
        ctx.strokeStyle = 'rgba(255,255,255,0.55)';
        ctx.lineWidth = 1.2;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    if (!st.f.e) {
      ctx.fillStyle = 'rgba(8,12,18,0.62)';
      ctx.fillRect(0, 0, g.cssW, g.cssH);
      ctx.fillStyle = '#e8eef6';
      ctx.font = '600 15px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Not a mappable wide view (replay, close-up or no calibration)', g.cssW / 2, g.cssH / 2);
      ctx.textAlign = 'start';
      return;
    }

    if ($('tTrails').checked) {
      const trails = new Map();
      for (let j = st.i; j >= 0 && frames[j].t >= t - 3 && frames[j].s === st.f.s; j--) {
        if (!frames[j].e) break;
        for (const r of frames[j].p) {
          if (!r[4]) continue;
          if (!trails.has(r[0])) trails.set(r[0], { kind: r[1], pts: [] });
          trails.get(r[0]).pts.push([r[2], r[3], frames[j].t]);
        }
      }
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      for (const { kind, pts } of trails.values()) {
        if (kind > 1 || pts.length < 2) continue;
        for (let j = 0; j < pts.length - 1; j++) {
          // A jump this large is a re-identification or cut, not movement.
          if (Math.hypot(pts[j][0] - pts[j + 1][0], pts[j][1] - pts[j + 1][1]) > 6) continue;
          const age = (t - pts[j][2]) / 3;
          ctx.strokeStyle = hexA(colours[KIND_TEAM[kind]], clamp(0.55 * (1 - age), 0, 0.55));
          ctx.beginPath();
          ctx.moveTo(X(pts[j][0]), Y(pts[j][1]));
          ctx.lineTo(X(pts[j + 1][0]), Y(pts[j + 1][1]));
          ctx.stroke();
        }
      }
      const ballPts = [];
      for (let j = st.i; j >= 0 && frames[j].t >= t - 3 && frames[j].s === st.f.s; j--) {
        if (frames[j].b && frames[j].b[2]) ballPts.push([frames[j].b[0], frames[j].b[1]]);
      }
      if (ballPts.length > 1) {
        ctx.strokeStyle = 'rgba(255,255,255,0.65)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ballPts.forEach(([x, y], j) => (j ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
        ctx.stroke();
      }
    }

    if (selected != null) drawSelectedPath(ctx, g, st);
    if ($('tVelocity').checked) {
      for (const p of st.players) {
        if (!p.o || (p.kind > 1 && p.kind !== 3)) continue;
        const v = velocity(st.i, p.id);
        if (!v || v.speed < 1.5) continue;
        arrow(ctx, X(p.x), Y(p.y), X(p.x + v.vx), Y(p.y + v.vy), 'rgba(255,255,255,0.85)', 1.6);
      }
    }
    const radius = Math.max(5, g.scale * 1.05);
    for (const p of st.players) {
      const x = X(p.x);
      const y = Y(p.y);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      let fill = '#8995a8';
      if (p.kind <= 1) fill = colours[KIND_TEAM[p.kind]];
      if (p.kind === 3) fill = '#d6f04d';
      if (p.kind === 4) fill = '#151a22';
      if (p.o) {
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = 1.8;
        ctx.strokeStyle = p.kind === 4 ? '#ffd23f' : 'rgba(255,255,255,0.92)';
        ctx.stroke();
      } else {
        ctx.fillStyle = hexA(fill.length === 7 ? fill : '#8995a8', 0.22);
        ctx.fill();
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = 'rgba(255,255,255,0.6)';
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (p.id === selected) {
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
        ctx.lineWidth = 2.4;
        ctx.strokeStyle = '#ffd23f';
        ctx.stroke();
      }
      if (p.kind <= 1 || p.kind === 3) {
        const isSel = p.id === selected;
        if (isSel || $('tLabels').checked) {
          ctx.font = `${isSel ? 700 : 600} ${isSel ? 12 : 10}px system-ui, sans-serif`;
          ctx.textAlign = 'center';
          ctx.lineWidth = 3;
          ctx.strokeStyle = 'rgba(8,12,18,0.85)';
          ctx.fillStyle = isSel ? '#ffd23f' : '#e8eef6';
          const label = isSel ? `${tag(p.id)} ${speedLabel(st, p.id)}` : tag(p.id);
          ctx.strokeText(label, x, y - radius - (isSel ? 8 : 4));
          ctx.fillText(label, x, y - radius - (isSel ? 8 : 4));
          ctx.textAlign = 'start';
        }
      }
    }
    if (st.ball) {
      const x = X(st.ball.x);
      const y = Y(st.ball.y);
      ctx.beginPath();
      ctx.arc(x, y, radius * 1.9, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.14)';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, Math.max(3.5, radius * 0.6), 0, Math.PI * 2);
      if (st.ball.o) {
        ctx.fillStyle = '#fff';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#0a0e14';
        ctx.stroke();
      } else {
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  function speedLabel(st, id) {
    const v = velocity(st.i, id);
    return v ? `${(v.speed * 3.6).toFixed(0)} km/h` : '';
  }

  /* ---------- video overlay ---------- */

  function drawOverlay(st) {
    const wrap = $('videoWrap');
    const ctx = sizeCanvas($('overlay'), wrap.clientWidth, wrap.clientHeight);
    ctx.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);
    if (!$('tBoxes').checked || !st.f.e) return;
    const sx = wrap.clientWidth / W;
    const sy = wrap.clientHeight / H;
    ctx.lineWidth = 2;
    for (const p of st.players) {
      if (!p.box || !p.o) continue;
      const colour = p.kind <= 1 ? colours[KIND_TEAM[p.kind]] : p.kind === 3 ? '#d6f04d' : p.kind === 4 ? '#ffd23f' : '#9aa6b8';
      const [x1, y1, x2, y2] = p.box;
      ctx.strokeStyle = p.id === selected ? '#ffd23f' : colour;
      ctx.lineWidth = p.id === selected ? 3.5 : 2;
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      if (p.id === selected) {
        const text = `${tag(p.id)} ${speedLabel(st, p.id)}`.trim();
        ctx.font = '700 12px system-ui, sans-serif';
        const w = ctx.measureText(text).width + 12;
        ctx.fillStyle = '#ffd23f';
        ctx.fillRect(x1 * sx, y1 * sy - 20, w, 18);
        ctx.fillStyle = '#0a0e14';
        ctx.fillText(text, x1 * sx + 6, y1 * sy - 7);
      }
    }
    ctx.lineWidth = 2;
    if (st.ball && st.ball.px != null && st.ball.o) {
      const x = st.ball.px * sx;
      const y = st.ball.py * sy;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(x, y, 11, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y - 20);
      ctx.lineTo(x - 5, y - 29);
      ctx.lineTo(x + 5, y - 29);
      ctx.closePath();
      ctx.fillStyle = '#fff';
      ctx.fill();
    }
  }

  /* ---------- momentum timeline ---------- */

  const chart = { padL: 84, padR: 14, padT: 40, padB: 112, yMax: 0.6 };

  function chartGeometry() {
    const canvas = $('momentum');
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    return {
      cssW,
      cssH,
      x0: chart.padL,
      x1: cssW - chart.padR,
      y0: chart.padT,
      y1: cssH - chart.padB,
    };
  }

  const timeToX = (g, t) => g.x0 + (t / D.source.duration) * (g.x1 - g.x0);
  const xToTime = (g, x) => clamp(((x - g.x0) / (g.x1 - g.x0)) * D.source.duration, 0, D.source.duration);

  function ellipsize(ctx, text, width) {
    if (ctx.measureText(text).width <= width) return text;
    let out = text;
    while (out.length > 2 && ctx.measureText(`${out}\u2026`).width > width) out = out.slice(0, -1);
    return `${out}\u2026`;
  }

  function drawMomentum(t) {
    const canvas = $('momentum');
    const g = chartGeometry();
    const ctx = sizeCanvas(canvas, g.cssW, g.cssH);
    ctx.clearRect(0, 0, g.cssW, g.cssH);
    const mid = (g.y0 + g.y1) / 2;
    const half = (g.y1 - g.y0) / 2;
    const Y = (v) => mid - (v / chart.yMax) * half;
    const b = D.bin_seconds;

    // Grid and axis.
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = '#8a9bb3';
    ctx.strokeStyle = 'rgba(138,155,179,0.18)';
    ctx.lineWidth = 1;
    const step = D.source.duration > 300 ? 60 : D.source.duration > 90 ? 15 : D.source.duration > 40 ? 10 : 5;
    ctx.textAlign = 'center';
    for (let s = 0; s <= D.source.duration + 1e-6; s += step) {
      const x = timeToX(g, s);
      ctx.beginPath();
      ctx.moveTo(x, g.y0);
      ctx.lineTo(x, g.y1);
      ctx.stroke();
      ctx.fillText(fmt(s), x, g.cssH - 5);
    }
    ctx.textAlign = 'right';
    for (const v of [-chart.yMax, 0, chart.yMax]) {
      ctx.beginPath();
      ctx.moveTo(g.x0, Y(v));
      ctx.lineTo(g.x1, Y(v));
      ctx.strokeStyle = v === 0 ? 'rgba(232,238,246,0.45)' : 'rgba(138,155,179,0.18)';
      ctx.stroke();
    }
    ctx.fillStyle = colours.A;
    ctx.fillText(`${names.A} ▲`, g.x0 - 8, Y(chart.yMax * 0.55));
    ctx.fillStyle = colours.B;
    ctx.fillText(`${names.B} ▼`, g.x0 - 8, Y(-chart.yMax * 0.55));

    // Build points with zero crossings so each team's area is exact.
    const points = [];
    const segments = [];
    let current = [];
    D.momentum.forEach((m) => {
      const x = timeToX(g, m.t + b / 2);
      if (m.value == null) {
        if (current.length) segments.push(current);
        current = [];
      } else {
        current.push({ x, v: clamp(m.value, -chart.yMax, chart.yMax), s: m.support });
      }
    });
    if (current.length) segments.push(current);

    for (const seg of segments) {
      const pts = [];
      for (let i = 0; i < seg.length; i++) {
        pts.push(seg[i]);
        const n = seg[i + 1];
        if (n && seg[i].v * n.v < 0) {
          const k = seg[i].v / (seg[i].v - n.v);
          pts.push({ x: lerp(seg[i].x, n.x, k), v: 0, s: Math.min(seg[i].s, n.s) });
        }
      }
      for (const sign of [1, -1]) {
        ctx.beginPath();
        ctx.moveTo(pts[0].x, Y(0));
        for (const p of pts) ctx.lineTo(p.x, Y(sign * Math.max(0, sign * p.v)));
        ctx.lineTo(pts[pts.length - 1].x, Y(0));
        ctx.closePath();
        const colour = sign > 0 ? colours.A : colours.B;
        const grad = ctx.createLinearGradient(0, mid, 0, sign > 0 ? g.y0 : g.y1);
        grad.addColorStop(0, hexA(colour, 0.18));
        grad.addColorStop(1, hexA(colour, 0.72));
        ctx.fillStyle = grad;
        ctx.fill();
      }
      ctx.lineWidth = 2;
      for (let i = 0; i < pts.length - 1; i++) {
        const sign = (pts[i].v + pts[i + 1].v) / 2 >= 0 ? 1 : -1;
        ctx.strokeStyle = sign > 0 ? colours.A : colours.B;
        ctx.setLineDash(Math.min(pts[i].s, pts[i + 1].s) < 0.4 ? [4, 4] : []);
        ctx.beginPath();
        ctx.moveTo(pts[i].x, Y(pts[i].v));
        ctx.lineTo(pts[i + 1].x, Y(pts[i + 1].v));
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // Gaps with no reliable signal.
    ctx.fillStyle = 'rgba(138,155,179,0.12)';
    let gapStart = null;
    D.momentum.forEach((m, i) => {
      const gap = m.value == null;
      if (gap && gapStart == null) gapStart = m.t;
      const end = i === D.momentum.length - 1 ? m.t + b : null;
      if ((!gap || end != null) && gapStart != null) {
        const stop = gap ? end : m.t;
        const x = timeToX(g, gapStart);
        const w = timeToX(g, stop) - x;
        ctx.fillRect(x, g.y0, w, g.y1 - g.y0);
        if (w > 90) {
          ctx.fillStyle = '#8a9bb3';
          ctx.textAlign = 'center';
          ctx.fillText('no reliable signal', x + w / 2, mid - 6);
          ctx.fillStyle = 'rgba(138,155,179,0.12)';
        }
        gapStart = null;
      }
    });

    // Strips: observed ball control, then mappable view.
    const laneY = g.y1 + 6;
    const laneH = 24;
    const peakTempo = Math.max(1, ...D.momentum.map((m) => m.intensity_mps || 0));
    ctx.fillStyle = 'rgba(138,155,179,0.08)';
    ctx.fillRect(g.x0, laneY, g.x1 - g.x0, laneH);
    ctx.textAlign = 'right';
    ctx.fillStyle = '#8a9bb3';
    ctx.fillText('tempo', g.x0 - 8, laneY + 15);
    ctx.beginPath();
    let open = false;
    D.momentum.forEach((m) => {
      if (m.intensity_mps == null) {
        open = false;
        return;
      }
      const x = timeToX(g, m.t + b / 2);
      const y = laneY + laneH - (m.intensity_mps / peakTempo) * (laneH - 3);
      if (open) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
      open = true;
    });
    ctx.strokeStyle = '#ffbf47';
    ctx.lineWidth = 1.6;
    ctx.stroke();
    const stripY = g.y1 + 38;
    ctx.textAlign = 'right';
    ctx.fillStyle = '#8a9bb3';
    ctx.fillText('control', g.x0 - 8, stripY + 6);
    ctx.fillText('view', g.x0 - 8, stripY + 18);
    const runs = (key, fn, y, hgt) => {
      let startT = null;
      let val = null;
      const flush = (endT) => {
        if (startT != null && val != null) {
          ctx.fillStyle = fn(val);
          ctx.fillRect(timeToX(g, startT), y, Math.max(1, timeToX(g, endT) - timeToX(g, startT)), hgt);
        }
      };
      for (const f of frames) {
        const v = key(f);
        if (v !== val) {
          flush(f.t);
          startT = f.t;
          val = v;
        }
      }
      flush(D.source.duration);
    };
    ctx.fillStyle = 'rgba(138,155,179,0.12)';
    ctx.fillRect(g.x0, stripY, g.x1 - g.x0, 6);
    ctx.fillRect(g.x0, stripY + 11, g.x1 - g.x0, 6);
    runs((f) => f.c, (v) => colours[v], stripY, 6);
    runs((f) => (f.e ? 1 : 0), (v) => (v ? 'rgba(70,224,160,0.7)' : 'rgba(255,107,107,0.75)'), stripY + 11, 6);

    // Space control lane: share of the camera-visible pitch nearest each team's observed players.
    const spaceY = g.y1 + 66;
    const spaceH = 22;
    const series = (D.space_control && D.space_control.series) || [];
    if (series.length) {
      ctx.fillStyle = 'rgba(138,155,179,0.08)';
      ctx.fillRect(g.x0, spaceY, g.x1 - g.x0, spaceH);
      ctx.textAlign = 'right';
      ctx.fillStyle = '#8a9bb3';
      ctx.fillText('space', g.x0 - 8, spaceY + 15);
      const mid50 = spaceY + spaceH / 2;
      series.forEach((row) => {
        if (row.A == null) return;
        const x0 = timeToX(g, row.t);
        const x1 = Math.max(x0 + 1, timeToX(g, row.t + 1));
        const hgt = (Math.abs(row.A - 0.5) / 0.5) * (spaceH / 2);
        ctx.fillStyle = hexA(row.A >= 0.5 ? colours.A : colours.B, 0.75);
        ctx.fillRect(x0, row.A >= 0.5 ? mid50 - hgt : mid50, x1 - x0, Math.max(1, hgt));
      });
      ctx.strokeStyle = 'rgba(232,238,246,0.35)';
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(g.x0, mid50);
      ctx.lineTo(g.x1, mid50);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Critical-moment markers on the top lane.
    D.moments.forEach((m) => {
      const x = timeToX(g, m.t);
      const x2 = timeToX(g, Math.max(m.end, m.t + 0.5));
      const colour = teamColour(m.team);
      ctx.strokeStyle = hexA(colour, 0.7);
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x, 30);
      ctx.lineTo(x2, 30);
      ctx.stroke();
      const active = t >= m.start - 0.5 && t <= Math.max(m.end, m.start + 3);
      ctx.beginPath();
      ctx.arc(x, 16, active ? 11 : 9, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
      if (active) {
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#fff';
        ctx.stroke();
      }
      ctx.fillStyle = '#0a0e14';
      ctx.font = '700 11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(GLYPH[m.type] || '•', x, 20);
    });

    // Playhead.
    const px = timeToX(g, t);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, g.y0 - 4);
    ctx.lineTo(px, g.y1 + (series.length ? 90 : 62));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(px - 5, g.y0 - 8);
    ctx.lineTo(px + 5, g.y0 - 8);
    ctx.lineTo(px, g.y0 - 1);
    ctx.closePath();
    ctx.fillStyle = '#fff';
    ctx.fill();
  }

  function pickMoment(g, x, y) {
    if (y > 34) return null;
    let best = null;
    let dist = 13;
    for (const m of D.moments) {
      const d = Math.abs(timeToX(g, m.t) - x);
      if (d < dist) {
        best = m;
        dist = d;
      }
    }
    return best;
  }

  function showTooltip(e) {
    const g = chartGeometry();
    const rect = $('momentum').getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const tip = $('tooltip');
    if (x < g.x0 || x > g.x1) {
      tip.hidden = true;
      return;
    }
    const moment = pickMoment(g, x, y);
    const t = xToTime(g, x);
    let html;
    if (moment) {
      html = `<b>${esc(moment.title)}</b><br>${fmt1(moment.t)} &middot; ${moment.confidence} confidence`;
    } else {
      const m = momentumAt(t);
      const who = m.value == null ? null : esc(m.value >= 0 ? names.A : names.B);
      html = `<b>${fmt1(t)}</b><br>` +
        (m.value == null
          ? 'No reliable momentum signal here'
          : `${who} +${Math.abs(m.value).toFixed(2)} &middot; support ${Math.round(m.support * 100)}%`);
      const space = spaceAt(t);
      if (space != null) html += `<br><span style="color:#8a9bb3">space control ${esc(names.A)} ${Math.round(space * 100)}% &middot; ${esc(names.B)} ${Math.round((1 - space) * 100)}%</span>`;
      const parts = Object.entries(m.components || {}).map(([k, v]) => `${k} ${v >= 0 ? '+' : ''}${v.toFixed(2)}`);
      if (parts.length) html += `<br><span style="color:#8a9bb3">${parts.join(' &middot; ')}</span>`;
    }
    tip.innerHTML = html;
    tip.hidden = false;
    const parent = $('momentum').parentElement.getBoundingClientRect();
    tip.style.left = `${clamp(e.clientX - parent.left + 14, 8, parent.width - 290)}px`;
    tip.style.top = `${e.clientY - parent.top + 16}px`;
  }

  /* ---------- panels ---------- */

  function buildMoments() {
    const list = $('moments');
    $('momentCount').textContent = `${D.moments.length} flagged`;
    list.innerHTML = '';
    if (!D.moments.length) {
      list.innerHTML = '<li class="empty">No moment cleared the evidence thresholds in this clip.</li>';
    }
    D.moments.forEach((m) => {
      const li = document.createElement('li');
      li.className = 'moment';
      li.tabIndex = 0;
      li.dataset.id = m.id;
      li.innerHTML = `
        <div class="glyph" style="background:${teamColour(m.team)}">${GLYPH[m.type] || '•'}</div>
        <div>
          <h3></h3>
          <div class="meta">${fmt1(m.t)}${m.end - m.start >= 1 ? ` &ndash; ${fmt1(m.end)}` : ''} &middot; score ${Math.round(m.score * 100)}</div>
          <ul></ul>
        </div>
        <span class="conf ${m.confidence}">${m.confidence}</span>`;
      li.querySelector('h3').textContent = m.title;
      const ul = li.querySelector('ul');
      m.evidence.forEach((line) => {
        const e = document.createElement('li');
        e.textContent = line;
        ul.appendChild(e);
      });
      (m.notes || []).forEach((id) => {
        const e = document.createElement('li');
        e.textContent = `Shape: ${D.tactics[id].text}`;
        ul.appendChild(e);
      });
      const go = () => seek(Math.max(0, m.start - 1));
      li.addEventListener('click', go);
      li.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') go();
      });
      list.appendChild(li);
    });
  }

  function updateMoments(t) {
    document.querySelectorAll('.moment').forEach((li) => {
      const m = D.moments[Number(li.dataset.id)];
      const active = t >= m.start - 0.5 && t <= Math.max(m.end, m.start + 3);
      li.classList.toggle('active', active);
    });
  }

  function updateShape(t) {
    const second = Math.floor(t);
    const html = ['A', 'B'].map((team) => {
      const rows = (D.shapes && D.shapes[team]) || [];
      const stats = D.shape_stats && D.shape_stats[team];
      const row = rows.find((r) => r.t === second) || rows.find((r) => Math.abs(r.t - second) <= 1);
      let body;
      let tag = '';
      if (!row || !stats) {
        body = '<small>withheld: needs 6+ observed outfield players in a calibrated view</small>';
      } else {
        const kind = row.depth <= Math.min(stats.depth_low, stats.depth_median - 3) ? 'compact' : row.depth >= Math.max(stats.depth_high, stats.depth_median + 3) ? 'stretched' : '';
        tag = kind ? `<span class="tag ${kind}">${kind}</span>` : '<span class="tag">typical</span>';
        body = `<small>${row.depth.toFixed(0)} m deep &times; ${row.width.toFixed(0)} m wide &middot; clip median depth ${stats.depth_median.toFixed(0)} m &middot; ${row.n} players</small>`;
      }
      return `<div class="shape-row"><i style="background:${colours[team]}"></i><div><b>${esc(names[team])}</b>${body}</div>${tag}</div>`;
    }).join('');
    if (html !== updateShape.last) {
      updateShape.last = html;
      $('shape').innerHTML = html;
    }
  }

  function updateNotes(t) {
    const active = D.tactics.filter((n) => n.start <= t && t <= n.end);
    const recent = D.tactics
      .filter((n) => n.end < t && t - n.end <= 6)
      .sort((a, b) => b.end - a.end)
      .slice(0, 2);
    const key = active.map((n) => n.id).join(',') + '|' + recent.map((n) => n.id).join(',');
    if (key === notesKey) return;
    notesKey = key;
    const host = $('notes');
    host.innerHTML = '';
    const add = (n, cls) => {
      const el = document.createElement('div');
      el.className = `note ${cls}`;
      el.style.borderLeftColor = teamColour(n.team);
      el.innerHTML = '<div class="kind"></div><p></p><small></small>';
      el.querySelector('.kind').textContent = n.kind.replace(/_/g, ' ');
      el.querySelector('p').textContent = n.text;
      el.querySelector('small').textContent =
        `${Object.entries(n.evidence).map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`).join(' · ')} · ${n.players_observed} players observed`;
      host.appendChild(el);
    };
    active.forEach((n) => add(n, 'live'));
    recent.forEach((n) => add(n, 'recent'));
    if (!active.length && !recent.length) {
      host.innerHTML = '<div class="empty">No notable shape or pressure change right now.</div>';
    }
  }

  function updateStatus(t, st) {
    const f = st.f;
    const ball = st.ball;
    const counts = [0, 0];
    st.players.forEach((p) => {
      if (p.kind <= 1 && p.o) counts[p.kind] += 1;
    });
    const chip = (cls, html) => `<span class="chip ${cls}">${html}</span>`;
    $('status').innerHTML = [
      f.e ? chip('on', 'Wide view mapped') : chip('off', 'Excluded view'),
      chip(ball && ball.o ? 'on' : '', `Ball <b>${ball ? (ball.o ? 'observed' : 'predicted') : 'not seen'}</b>`),
      chip(f.c ? 'on' : '', `Control <b>${f.c ? esc(names[f.c]) : 'unknown'}</b>`),
      chip('', `<b style="color:${colours.A}">${counts[0]}</b> v <b style="color:${colours.B}">${counts[1]}</b> observed`),
    ].join('');
    $('videoState').className = `video-state${f.e ? '' : ' on'}`;
    $('videoState').textContent = f.e ? '' : 'Replay or close-up: not mapped';
    const m = momentumAt(t);
    $('now').innerHTML = m.value == null
      ? 'Momentum: <b>no reliable signal</b>'
      : `Momentum: <b style="color:${m.value >= 0 ? colours.A : colours.B}">${esc(m.value >= 0 ? names.A : names.B)} ${m.value >= 0 ? '+' : '−'}${Math.abs(m.value).toFixed(2)}</b> &middot; support ${Math.round(m.support * 100)}%`;
  }

  function buildStatic() {
    $('teams').innerHTML = ['A', 'B']
      .map((k) => `<div class="team-pill"><i style="background:${colours[k]}"></i>${esc(names[k])}</div>`)
      .join('');
    $('legend').innerHTML = [
      `<span><i style="background:${colours.A}"></i>${esc(names.A)} ahead</span>`,
      `<span><i style="background:${colours.B}"></i>${esc(names.B)} ahead</span>`,
      '<span><i style="background:rgba(138,155,179,.4)"></i>no reliable signal</span>',
    ].join('');
    const c = D.coverage;
    const meter = (label, value, text) =>
      `<div class="meter"><div><span>${label}</span><b>${text}</b></div><span class="bar"><i style="width:${clamp(value, 0, 1) * 100}%"></i></span></div>`;
    const pct = (v) => `${Math.round(v * 100)}%`;
    $('coverage').innerHTML = [
      meter('Mappable wide view', c.eligible_fraction, pct(c.eligible_fraction)),
      meter('Ball detected', c.ball_observed_fraction, pct(c.ball_observed_fraction)),
      meter('Ball passes sanity gate', c.ball_usable_fraction, pct(c.ball_usable_fraction)),
      meter('Team control known', c.control_covered_fraction, pct(c.control_covered_fraction)),
      meter('Momentum supported', c.momentum_supported_fraction, pct(c.momentum_supported_fraction)),
      meter('Outfield players observed', c.mean_observed_players / 20, `${c.mean_observed_players.toFixed(1)} of 20`),
    ].join('');
    $('limits').innerHTML = D.limitations.map((l) => `<li>${esc(l)}</li>`).join('');
    const foot = [];
    if (D.source.hardware) foot.push(`computed on ${D.source.hardware}`);
    if (D.source.samples_per_compute_second) foot.push(`${D.source.samples_per_compute_second.toFixed(2)} sampled frames/s`);
    foot.push(`${D.source.sample_hz} Hz samples`);
    foot.push(`model ${D.source.model_sha256.slice(0, 8)}`);
    $('pitchFoot').textContent = foot.join(' · ');
  }

  /* ---------- summary tiles, players, detector ---------- */

  const pct0 = (v) => `${Math.round(v * 100)}%`;
  const swatch = (team) => `<i class="sw" style="background:${team ? colours[team] : '#d6f04d'}"></i>`;

  function splitBar(a) {
    return `<span class="split"><i style="width:${a * 100}%;background:${colours.A}"></i><i style="width:${(1 - a) * 100}%;background:${colours.B}"></i></span>`;
  }

  function buildKpis() {
    const tiles = [];
    const tile = (label, value, sub, extra = '') =>
      tiles.push(`<div class="kpi"><span class="kpi-label">${label}</span><b>${value}</b>${extra}<small>${sub}</small></div>`);
    const poss = D.summary && D.summary.observed_possession_percent;
    if (poss) {
      const a = poss.A / 100;
      tile(
        'Observed possession',
        `<span style="color:${colours.A}">${Math.round(poss.A)}%</span> <em>/</em> <span style="color:${colours.B}">${Math.round(poss.B)}%</span>`,
        `only ${pct0(D.summary.possession_coverage)} of the clip had a clear ball owner`,
        splitBar(a),
      );
    }
    const sc = D.space_control;
    if (sc && sc.mean_A != null) {
      tile(
        'Space control',
        `<span style="color:${colours.A}">${Math.round(sc.mean_A * 100)}%</span> <em>/</em> <span style="color:${colours.B}">${Math.round((1 - sc.mean_A) * 100)}%</span>`,
        'of the camera-visible pitch nearest each team',
        splitBar(sc.mean_A),
      );
    }
    const bins = D.momentum.filter((m) => m.value != null && Math.abs(m.value) >= 0.05);
    if (bins.length) {
      const a = bins.filter((m) => m.value > 0).length / bins.length;
      tile(
        'Momentum ahead',
        `<span style="color:${colours.A}">${Math.round(a * 100)}%</span> <em>/</em> <span style="color:${colours.B}">${Math.round((1 - a) * 100)}%</span>`,
        `${bins.length} of ${D.momentum.length} seconds with a clear lead`,
        splitBar(a),
      );
    }
    if (playerList.length) {
      const outfield = playerList.filter((p) => p.role === 'player');
      const fastest = outfield.slice().sort((x, y) => y.peak_kmh - x.peak_kmh)[0];
      const far = outfield[0];
      tile('Players tracked', String(playerList.length), `${D.coverage.mean_observed_players.toFixed(1)} outfield players seen per frame`);
      if (fastest) {
        tile('Fastest', `${fastest.peak_kmh.toFixed(0)} <em>km/h</em>`, `${tag(fastest.id)} ${fastest.team ? esc(names[fastest.team]) : ''} &middot; 1 s average`);
      }
      if (far) {
        tile('Most ground', `${far.distance_m} <em>m</em>`, `${tag(far.id)} ${far.team ? esc(names[far.team]) : ''} &middot; ${far.seconds.toFixed(0)} s visible`);
      }
    }
    $('kpis').innerHTML = tiles.join('');
    $('kpis').hidden = !tiles.length;
  }

  const COLUMNS = [
    ['id', 'Player', false],
    ['seconds', 'Seen (s)', true],
    ['distance_m', 'Distance (m)', true],
    ['avg_kmh', 'Avg km/h', true],
    ['peak_kmh', 'Peak km/h', true],
    ['high_speed_s', '&gt;20 km/h (s)', true],
    ['sprint_s', '&gt;25 km/h (s)', true],
  ];

  function visiblePlayers() {
    const team = $('playerTeam').value;
    const rows = playerList.filter((p) => (team ? p.team === team : true));
    rows.sort((x, y) => (x[sortKey] > y[sortKey] ? 1 : x[sortKey] < y[sortKey] ? -1 : 0) * sortDir);
    return rows;
  }

  function renderPlayers() {
    const head = COLUMNS.map(
      ([key, label]) => `<th data-key="${key}" class="${key === sortKey ? 'sorted' : ''}" aria-sort="${key === sortKey ? (sortDir < 0 ? 'descending' : 'ascending') : 'none'}"><button type="button">${label}${key === sortKey ? (sortDir < 0 ? ' ▾' : ' ▴') : ''}</button></th>`,
    ).join('');
    $('players').tHead.innerHTML = `<tr>${head}</tr>`;
    const maxDistance = Math.max(1, ...playerList.map((p) => p.distance_m));
    $('players').tBodies[0].innerHTML = visiblePlayers()
      .map(
        (p) => `<tr data-id="${p.id}" class="${p.id === selected ? 'sel' : ''}" tabindex="0">
          <td>${swatch(p.team)}<b>${tag(p.id)}</b>${p.role === 'goalkeeper' ? ' <span class="gk">GK</span>' : ''}</td>
          <td>${p.seconds.toFixed(0)}</td>
          <td class="bar-cell" style="--w:${(p.distance_m / maxDistance) * 100}%"><span>${p.distance_m}</span></td>
          <td>${p.avg_kmh.toFixed(1)}</td>
          <td>${p.peak_kmh.toFixed(1)}</td>
          <td>${p.high_speed_s.toFixed(1)}</td>
          <td>${p.sprint_s.toFixed(1)}</td>
        </tr>`,
      )
      .join('');
  }

  function buildPlayers() {
    playerList = D.players || [];
    playerById = new Map(playerList.map((p) => [p.id, p]));
    if (!playerList.length) return;
    $('playersRow').hidden = false;
    $('playerTeam').insertAdjacentHTML(
      'beforeend',
      ['A', 'B'].map((k) => `<option value="${k}">${esc(names[k])}</option>`).join(''),
    );
    $('playerTeam').addEventListener('change', renderPlayers);
    $('players').tHead.addEventListener('click', (e) => {
      const th = e.target.closest('th');
      if (!th) return;
      const key = th.dataset.key;
      sortDir = key === sortKey ? -sortDir : key === 'id' ? 1 : -1;
      sortKey = key;
      renderPlayers();
    });
    const pick = (e) => {
      const row = e.target.closest('tr[data-id]');
      if (row) selectPlayer(Number(row.dataset.id));
    };
    $('players').tBodies[0].addEventListener('click', pick);
    $('players').tBodies[0].addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        pick(e);
      }
    });
    $('playerNote').textContent =
      'Distance and speed count only the seconds a player is visible on the broadcast camera, using smoothed pitch positions. ' +
      'Peak is a one-second average. A track can split in two when the tracker loses a player, so totals are lower bounds.';
    renderPlayers();
    updateSelectedUi();
  }

  function selectPlayer(id) {
    selected = id === selected ? null : id;
    renderPlayers();
    updateSelectedUi();
    const row = document.querySelector(`#players tr[data-id="${selected}"]`);
    if (row) row.scrollIntoView({ block: 'nearest' });
    render(true);
  }

  function updateSelectedUi() {
    const info = selected != null ? playerById.get(selected) : null;
    const chip = $('selectedChip');
    chip.hidden = !info;
    if (info) {
      chip.innerHTML = `${swatch(info.team)}<b>${tag(info.id)}</b> ${info.team ? esc(names[info.team]) : 'Goalkeeper'} &middot; ${info.distance_m} m &middot; peak ${info.peak_kmh.toFixed(0)} km/h <button type="button" id="clearSel">clear</button>`;
      $('clearSel').addEventListener('click', () => selectPlayer(selected));
    }
    $('playerDetail').innerHTML = info
      ? `<div class="detail-head">${swatch(info.team)}<b>${tag(info.id)}</b> <span>visible ${fmt(info.first)}&ndash;${fmt(info.last)} &middot; speed per second</span></div><canvas id="spark" aria-label="Speed of the selected player over time"></canvas>`
      : '<div class="empty">Select a player to follow them on the pitch and in the video, and to see their speed over time.</div>';
  }

  function drawSpark(t) {
    const canvas = $('spark');
    const info = selected != null ? playerById.get(selected) : null;
    if (!canvas || !info) return;
    const cssW = canvas.clientWidth;
    const cssH = 84;
    canvas.style.height = `${cssH}px`;
    const ctx = sizeCanvas(canvas, cssW, cssH);
    ctx.clearRect(0, 0, cssW, cssH);
    const data = info.kmh_per_s;
    const top = Math.max(30, ...data.map((v) => v || 0)) * 1.05;
    const X = (sec) => 30 + (sec / D.source.duration) * (cssW - 36);
    const Y = (v) => cssH - 16 - (v / top) * (cssH - 24);
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'right';
    for (const [v, label] of [[0, '0'], [20, '20'], [25, '']]) {
      ctx.strokeStyle = v ? 'rgba(255,191,71,0.35)' : 'rgba(138,155,179,0.25)';
      ctx.setLineDash(v ? [3, 3] : []);
      ctx.beginPath();
      ctx.moveTo(30, Y(v));
      ctx.lineTo(cssW - 6, Y(v));
      ctx.stroke();
      ctx.fillStyle = '#8a9bb3';
      ctx.fillText(label, 26, Y(v) + 3);
    }
    ctx.setLineDash([]);
    ctx.strokeStyle = info.team ? colours[info.team] : '#d6f04d';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let open = false;
    data.forEach((v, sec) => {
      if (v == null) {
        open = false;
        return;
      }
      if (open) ctx.lineTo(X(sec + 0.5), Y(v));
      else ctx.moveTo(X(sec + 0.5), Y(v));
      open = true;
    });
    ctx.stroke();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(X(t), 4);
    ctx.lineTo(X(t), cssH - 14);
    ctx.stroke();
    ctx.fillStyle = '#8a9bb3';
    ctx.textAlign = 'center';
    ctx.fillText('km/h', 14, 10);
  }

  function buildModel() {
    const m = D.model;
    if (!m) return;
    $('modelCard').hidden = false;
    $('modelSub').textContent = `${m.image_size ? `${m.image_size} px · ` : ''}${D.source.model_sha256.slice(0, 8)}`;
    const stat = (label, v) => `<div><b>${v.toFixed(3)}</b><span>${label}</span></div>`;
    const rows = m.classes
      .map(
        (c) => `<div class="cls ${c.name === 'ball' ? 'weak' : ''}"><span>${esc(c.name)} <small>${c.instances} in val</small></span><span class="bar"><i style="width:${c.recall * 100}%"></i></span><b>${Math.round(c.recall * 100)}%</b></div>`,
      )
      .join('');
    const ball = m.classes.find((c) => c.name === 'ball');
    $('model').innerHTML = `
      <div class="stats4">${stat('mAP50', m.map50)}${stat('mAP50-95', m.map50_95)}${stat('precision', m.precision)}${stat('recall', m.recall)}</div>
      <div class="cls-head"><span>Recall by class</span></div>
      ${rows}
      <p class="fine">${esc(m.split)}. A small, single-source validation set, so treat these as optimistic: there is no independent test on other leagues, cameras or lighting.${ball ? ` The ball is found in ${Math.round(ball.recall * 100)}% of validation images where it appears; when it is missed the page falls back to the camera focus and never counts it as observed.` : ''}</p>`;
  }

  let spaceKey = '';
  function updateSpaceLive(t) {
    const a = spaceAt(t);
    const key = a == null ? 'none' : String(Math.round(a * 100));
    if (key === spaceKey) return;
    spaceKey = key;
    $('spaceLive').innerHTML = a == null
      ? '<span class="muted">Space control withheld here: needs 3+ observed players per team in a mapped view.</span>'
      : `<span class="muted">Space control</span>${splitBar(a)}<b style="color:${colours.A}">${Math.round(a * 100)}%</b><b style="color:${colours.B}">${Math.round((1 - a) * 100)}%</b>`;
  }

  function pitchPick(e) {
    if (!lastSt) return;
    const canvas = $('pitch');
    const g = pitchGeometry(canvas);
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / g.scale - g.pad;
    const my = (e.clientY - rect.top) / g.scale - g.pad;
    let best = null;
    let dist = 3;
    for (const p of lastSt.players) {
      if ((p.kind > 1 && p.kind !== 3) || !playerById.has(p.id)) continue;
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < dist) {
        best = p;
        dist = d;
      }
    }
    if (best) selectPlayer(best.id);
  }

  /* ---------- control ---------- */

  function seek(t) {
    video.currentTime = clamp(t, 0, D.source.duration);
    render(true);
  }

  function render(force) {
    const t = video.currentTime;
    const wrap = $('videoWrap');
    const key = [$('tBoxes').checked, $('tTrails').checked, $('tFootprint').checked, $('tHeat').value, $('tSpace').checked, $('tVelocity').checked, $('tLabels').checked, selected].join();
    if (!force && t === last.t && wrap.clientWidth === last.w && window.innerWidth === last.h && key === last.key) return;
    last = { t, w: wrap.clientWidth, h: window.innerWidth, key };
    const st = stateAt(t);
    lastSt = st;
    drawOverlay(st);
    drawPitch(t, st);
    drawMomentum(t);
    updateStatus(t, st);
    updateMoments(t);
    updateShape(t);
    updateNotes(t);
    updateSpaceLive(t);
    drawSpark(t);
    $('time').textContent = `${fmt(t)} / ${fmt(D.source.duration)}`;
    if (document.activeElement !== $('scrub')) $('scrub').value = Math.round((t / D.source.duration) * 1000);
    $('play').innerHTML = video.paused ? '&#9654;' : '&#10073;&#10073;';
  }

  function loop() {
    render(false);
    requestAnimationFrame(loop);
  }

  function togglePlay() {
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  }

  function jumpMoment(dir) {
    const t = video.currentTime;
    const ordered = D.moments.slice().sort((a, b) => a.start - b.start);
    const target =
      dir > 0
        ? ordered.find((m) => m.start - 1 > t + 0.1)
        : ordered.reverse().find((m) => m.start - 1 < t - 0.6);
    if (target) seek(Math.max(0, target.start - 1));
  }

  function bind() {
    $('play').addEventListener('click', togglePlay);
    video.addEventListener('click', togglePlay);
    video.addEventListener('play', () => render(true));
    video.addEventListener('pause', () => render(true));
    video.addEventListener('seeked', () => render(true));
    $('scrub').addEventListener('input', (e) => seek((Number(e.target.value) / 1000) * D.source.duration));
    $('rate').addEventListener('change', (e) => {
      video.playbackRate = Number(e.target.value);
    });
    $('tHeat').addEventListener('change', (e) => {
      computeHeat(e.target.value);
      render(true);
    });
    ['tBoxes', 'tTrails', 'tFootprint', 'tSpace', 'tVelocity', 'tLabels'].forEach((id) => $(id).addEventListener('change', () => render(true)));
    $('pitch').addEventListener('click', pitchPick);
    const chartEl = $('momentum');
    let dragging = false;
    const scrubTo = (e) => {
      const g = chartGeometry();
      const rect = chartEl.getBoundingClientRect();
      seek(xToTime(g, e.clientX - rect.left));
    };
    chartEl.addEventListener('pointerdown', (e) => {
      const g = chartGeometry();
      const rect = chartEl.getBoundingClientRect();
      const moment = pickMoment(g, e.clientX - rect.left, e.clientY - rect.top);
      if (moment) {
        seek(Math.max(0, moment.start - 1));
        return;
      }
      dragging = true;
      chartEl.setPointerCapture(e.pointerId);
      scrubTo(e);
    });
    chartEl.addEventListener('pointermove', (e) => {
      if (dragging) scrubTo(e);
      showTooltip(e);
    });
    chartEl.addEventListener('pointerup', () => {
      dragging = false;
    });
    chartEl.addEventListener('pointerleave', () => {
      $('tooltip').hidden = true;
    });
    window.addEventListener('keydown', (e) => {
      if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) && e.target.type !== 'range') return;
      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.key === 'ArrowLeft') seek(video.currentTime - (e.shiftKey ? 5 : 1));
      else if (e.key === 'ArrowRight') seek(video.currentTime + (e.shiftKey ? 5 : 1));
      else if (e.key === ']') jumpMoment(1);
      else if (e.key === '[') jumpMoment(-1);
    });
    window.addEventListener('resize', () => render(true));
  }

  async function main() {
    let response;
    try {
      response = await fetch(`${base}analysis.json`, { cache: 'no-cache' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      D = await response.json();
    } catch (error) {
      const box = $('error');
      box.hidden = false;
      // Built from text nodes: the data path comes from the URL.
      box.textContent = '';
      const add = (tag, text) => {
        const el = document.createElement(tag);
        el.textContent = text;
        box.appendChild(el);
        return el;
      };
      add('b', 'Could not load the analysis.');
      add('p', error.message);
      const hint = add('p', `This page fetches ${base}analysis.json. Serve the folder over HTTP, for example `);
      const code = document.createElement('code');
      code.textContent = 'python -m http.server -d web';
      hint.append(code, ', rather than opening the file directly.');
      return;
    }
    if (D.schema_version !== 1) {
      $('error').hidden = false;
      $('error').textContent = `Unsupported analysis schema ${D.schema_version}.`;
      return;
    }
    frames = D.frames;
    [W, H] = D.source.image_size;
    L = D.pitch.length;
    PW = D.pitch.width;
    names = { A: D.teams.A.name, B: D.teams.B.name };
    colours = { A: D.teams.A.colour, B: D.teams.B.colour };
    document.documentElement.style.setProperty('--team-a', colours.A);
    document.documentElement.style.setProperty('--team-b', colours.B);
    const peak = Math.max(0.3, ...D.momentum.map((m) => (m.value == null ? 0 : Math.abs(m.value))));
    chart.yMax = clamp(peak * 1.2, 0.5, 1);

    $('app').hidden = false;
    $('videoWrap').style.aspectRatio = `${W} / ${H}`;
    video = $('video');
    for (const source of D.source.videos || [{ src: D.source.video, type: 'video/mp4' }]) {
      const el = document.createElement('source');
      el.src = `${base}${source.src}`;
      el.type = source.type;
      video.appendChild(el);
    }
    video.load();
    buildStatic();
    buildPlayers();
    buildKpis();
    buildModel();
    buildMoments();
    bind();
    const start = Number(params.get('t'));
    video.addEventListener('loadedmetadata', () => {
      if (start > 0) video.currentTime = start;
      if (params.get('autoplay') === '1') video.play().catch(() => {});
      render(true);
    }, { once: true });
    window.__viewer = { seek, togglePlay, get duration() { return D.source.duration; } };
    loop();
  }

  main();
})();
