// 計画T（11/27）：郡司研の YouTube の動画から、個体の位置を取り出して測る（ブラウザの開発者ツールで、動画のページで実行する）。
// 動画の1マスは 2 画素（個体は 3×3 画素の黒い四角、隣どうしで 1 画素重なる）。場は 100×100 マス（どの動画も同じ）。
// 動画は画面を録画したもので、個体を描き直している途中のフレーム（25〜30 体ずつ増える）が混ざる。
// そのため、個体の数が一番多いフレームの 97〜103% のフレームだけを使い、続けて同じフレームは1つにまとめて「1ステップ」とした。
// 使い方：await __grab(開始の秒, フレーム数) で位置を取り出し、__anal(その結果) で測る。
window.__grab = async (t0, n) => {
  const v = document.querySelector('video'); v.muted = true; v.pause();
  const c = document.createElement('canvas'); const out = [];
  for (let k = 0; k < n; k++) {
    const t = t0 + k / 30 + 0.005; v.currentTime = t;
    await new Promise(r => { const f = () => { v.removeEventListener('seeked', f); r(); }; v.addEventListener('seeked', f); setTimeout(r, 2000); });
    c.width = v.videoWidth; c.height = v.videoHeight; const g = c.getContext('2d'); g.drawImage(v, 0, 0);
    const W = c.width, H = c.height, d = g.getImageData(0, 0, W, H).data;
    const dk = (x, y) => x >= 0 && y >= 0 && x < W && y < H && d[(y * W + x) * 4] < 100;
    let best = null;
    for (const ox of [0, 1]) for (const oy of [0, 1]) {          // マスの中心が奇数・偶数のどちらの画素にあるか
      const cl = [];
      for (let cy = oy + 1; cy < H - 1; cy += 2) for (let cx = ox + 1; cx < W - 1; cx += 2) {
        let s = 0; for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) s += dk(cx + a, cy + b);
        if (s >= 8) cl.push(((cx - 1 - ox) / 2) + ',' + ((cy - 1 - oy) / 2));
      }
      if (!best || cl.length > best.length) best = cl;
    }
    out.push(t.toFixed(3) + '|' + best.join(';'));
  }
  return out;
};

window.__anal = (raw) => {
  const S = 100;
  const fr = raw.map(s => { const [t, c] = s.split('|'); return { t: +t, c: c ? c.split(';').map(p => p.split(',').map(Number)) : [] }; });
  const pd = (a, b) => { let dx = b[0] - a[0], dy = b[1] - a[1]; dx -= S * Math.round(dx / S); dy -= S * Math.round(dy / S); return [dx, dy]; };
  const key = c => c.map(p => p.join(',')).sort().join(';');
  const Nmax = Math.max(...fr.map(f => f.c.length));
  const full = fr.filter(f => f.c.length >= 0.97 * Nmax && f.c.length <= 1.03 * Nmax); if (!full.length) return { Nmax };
  const uq = [full[0]]; for (const f of full.slice(1)) if (key(f.c) !== key(uq[uq.length - 1].c)) uq.push(f);
  let iso = 0, stay = 0, lsum = 0, all = 0, allStay = 0; const lh = {}; const pols = [];
  for (let i = 1; i < uq.length; i++) {
    const A = uq[i - 1].c, B = uq[i].c; let px = 0, py = 0, pn = 0;
    for (const a of A) {
      let bd = 1e9, bb = null; for (const b of B) { const d = pd(a, b); const l = Math.hypot(d[0], d[1]); if (l < bd) { bd = l; bb = d; } }   // 次のフレームの一番近い個体
      if (bd > 0) { px += bb[0] / bd; py += bb[1] / bd; } pn++; all++; if (bd === 0) allStay++;
      let near = 1e9; for (const o of A) { if (o === a) continue; const d = pd(a, o); near = Math.min(near, Math.hypot(d[0], d[1])); }
      if (near >= 8 && bd < 7) { iso++; if (bd === 0) stay++; lsum += bd; const k = Math.round(bd); lh[k] = (lh[k] || 0) + 1; }   // 離れた個体だけ（取り違えが起きにくい）
    }
    pols.push(Math.hypot(px, py) / pn);
  }
  const A = uq[uq.length - 1].c; const seen = new Array(A.length).fill(false); const comps = [];
  for (let i = 0; i < A.length; i++) {
    if (seen[i]) continue; const st = [i]; seen[i] = true; let sz = 0;
    while (st.length) { const j = st.pop(); sz++; for (let k = 0; k < A.length; k++) { if (seen[k]) continue; const d = pd(A[j], A[k]); if (Math.max(Math.abs(d[0]), Math.abs(d[1])) <= 2) { seen[k] = true; st.push(k); } } }
    comps.push(sz);
  }
  comps.sort((a, b) => b - a);
  const span = uq[uq.length - 1].t - uq[0].t;
  return { t0: fr[0].t, N: Nmax, steps: uq.length - 1, sps: +((uq.length - 1) / span).toFixed(1), iso,
    stayIso: +(stay / Math.max(iso, 1)).toFixed(2), stayAll: +(allStay / Math.max(all, 1)).toFixed(2),
    lenIso: +(lsum / Math.max(iso, 1)).toFixed(2), lh: JSON.stringify(lh),
    pol: +(pols.reduce((a, b) => a + b, 0) / Math.max(pols.length, 1)).toFixed(2), clusters: comps.length, big: comps.slice(0, 5).join(',') };
};
