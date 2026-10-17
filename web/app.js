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

  const fmt = (t) => {
    t = Math.max(0, t);
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
  };
  const fmt1 = (t) => `${fmt(t)}.${Math.floor((t % 1) * 10)}`;
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

  function drawPitch(t, st) {
    const canvas = $('pitch');
    const g = pitchGeometry(canvas);
    canvas.style.height = `${g.cssH}px`;
    const ctx = sizeCanvas(canvas, g.cssW, g.cssH);
    const X = (x) => (g.pad + x) * g.scale;
    const Y = (y) => (g.pad + y) * g.scale;
    drawPitchMarkings(ctx, g);
    if (heat) drawHeat(ctx, g);

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
      ctx.strokeStyle = colour;
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    }
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

  const chart = { padL: 84, padR: 14, padT: 40, padB: 84, yMax: 0.6 };

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
    ctx.lineTo(px, g.y1 + 62);
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
      html = `<b>${moment.title}</b><br>${fmt1(moment.t)} &middot; ${moment.confidence} confidence`;
    } else {
      const m = momentumAt(t);
      const who = m.value == null ? null : m.value >= 0 ? names.A : names.B;
      html = `<b>${fmt1(t)}</b><br>` +
        (m.value == null
          ? 'No reliable momentum signal here'
          : `${who} +${Math.abs(m.value).toFixed(2)} &middot; support ${Math.round(m.support * 100)}%`);
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
      return `<div class="shape-row"><i style="background:${colours[team]}"></i><div><b>${names[team]}</b>${body}</div>${tag}</div>`;
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
      chip(f.c ? 'on' : '', `Control <b>${f.c ? names[f.c] : 'unknown'}</b>`),
      chip('', `<b style="color:${colours.A}">${counts[0]}</b> v <b style="color:${colours.B}">${counts[1]}</b> observed`),
    ].join('');
    $('videoState').className = `video-state${f.e ? '' : ' on'}`;
    $('videoState').textContent = f.e ? '' : 'Replay or close-up: not mapped';
    const m = momentumAt(t);
    $('now').innerHTML = m.value == null
      ? 'Momentum: <b>no reliable signal</b>'
      : `Momentum: <b style="color:${m.value >= 0 ? colours.A : colours.B}">${m.value >= 0 ? names.A : names.B} ${m.value >= 0 ? '+' : '−'}${Math.abs(m.value).toFixed(2)}</b> &middot; support ${Math.round(m.support * 100)}%`;
  }

  function buildStatic() {
    $('teams').innerHTML = ['A', 'B']
      .map((k) => `<div class="team-pill"><i style="background:${colours[k]}"></i>${names[k]}</div>`)
      .join('');
    $('legend').innerHTML = [
      `<span><i style="background:${colours.A}"></i>${names.A} ahead</span>`,
      `<span><i style="background:${colours.B}"></i>${names.B} ahead</span>`,
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
    $('limits').innerHTML = D.limitations.map((l) => `<li>${l}</li>`).join('');
    const foot = [];
    if (D.source.hardware) foot.push(`computed on ${D.source.hardware}`);
    if (D.source.samples_per_compute_second) foot.push(`${D.source.samples_per_compute_second.toFixed(2)} sampled frames/s`);
    foot.push(`${D.source.sample_hz} Hz samples`);
    foot.push(`model ${D.source.model_sha256.slice(0, 8)}`);
    $('pitchFoot').textContent = foot.join(' · ');
  }

  /* ---------- control ---------- */

  function seek(t) {
    video.currentTime = clamp(t, 0, D.source.duration);
    render(true);
  }

  function render(force) {
    const t = video.currentTime;
    const wrap = $('videoWrap');
    const key = [$('tBoxes').checked, $('tTrails').checked, $('tFootprint').checked, $('tHeat').value].join();
    if (!force && t === last.t && wrap.clientWidth === last.w && window.innerWidth === last.h && key === last.key) return;
    last = { t, w: wrap.clientWidth, h: window.innerWidth, key };
    const st = stateAt(t);
    drawOverlay(st);
    drawPitch(t, st);
    drawMomentum(t);
    updateStatus(t, st);
    updateMoments(t);
    updateShape(t);
    updateNotes(t);
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
    ['tBoxes', 'tTrails', 'tFootprint'].forEach((id) => $(id).addEventListener('change', () => render(true)));
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
      box.innerHTML = `<b>Could not load the analysis.</b><br>${error.message}<br><br>This page fetches <code>${base}analysis.json</code>. Serve the folder over HTTP, for example <code>python -m http.server -d web</code>, rather than opening the file directly.`;
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
