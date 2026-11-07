// Procesadores de audio en tiempo real (AudioWorklet).
// Todos son de latencia cero, para que el stem exportado quede alineado al
// milisegundo con el resto de stems de la canción.

const LOG10 = Math.log(10);
const toDb = (x) => 20 * Math.log(x + 1e-12) / LOG10;
const fromDb = (d) => Math.exp(d * LOG10 / 20);
const coef = (ms) => Math.exp(-1 / Math.max(1, (ms / 1000) * sampleRate));
const kp = (name, def, min, max) => ({ name, defaultValue: def, minValue: min, maxValue: max, automationRate: 'k-rate' });

function passThrough(inp, out) {
  for (let c = 0; c < out.length; c++) {
    const src = inp[Math.min(c, inp.length - 1)];
    if (src) out[c].set(src);
  }
}

// ---------------------------------------------------------------------------
// Puerta de ruido suave = expansor descendente con rango limitado
// ---------------------------------------------------------------------------
class GateProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [kp('enabled', 1, 0, 1), kp('threshold', -60, -100, 0), kp('ratio', 2, 1, 10),
      kp('range', 15, 0, 60), kp('attack', 2, 0.05, 100), kp('release', 120, 5, 3000), kp('hold', 20, 0, 1000)];
  }
  constructor() { super(); this.env = 0; this.g = 1; this.holdCnt = 0; this.cnt = 0; this.minG = 1; }
  process(inputs, outputs, p) {
    const inp = inputs[0], out = outputs[0];
    if (!inp || inp.length === 0) return true;
    if (p.enabled[0] < 0.5) { passThrough(inp, out); this.g = 1; return true; }
    const n = inp[0].length, nc = inp.length;
    const thr = p.threshold[0], ratio = p.ratio[0], range = p.range[0];
    const eA = coef(0.5), eR = coef(30), gA = coef(p.attack[0]), gR = coef(p.release[0]);
    const holdS = (p.hold[0] / 1000) * sampleRate;
    let env = this.env, g = this.g, hc = this.holdCnt, minG = this.minG;
    for (let i = 0; i < n; i++) {
      let a = 0;
      for (let c = 0; c < nc; c++) { const v = Math.abs(inp[c][i]); if (v > a) a = v; }
      env = a > env ? eA * env + (1 - eA) * a : eR * env + (1 - eR) * a;
      const lvl = toDb(env);
      let target = 1;
      if (lvl >= thr) hc = holdS;
      else if (hc > 0) hc--;
      else target = fromDb(Math.max(-(thr - lvl) * (ratio - 1), -range));
      g = target > g ? gA * g + (1 - gA) * target : gR * g + (1 - gR) * target;
      if (g < minG) minG = g;
      for (let c = 0; c < out.length; c++) out[c][i] = inp[Math.min(c, nc - 1)][i] * g;
    }
    this.env = env; this.g = g; this.holdCnt = hc;
    if (++this.cnt >= 16) { this.port.postMessage(toDb(minG)); this.cnt = 0; minG = 1; }
    this.minG = minG;
    return true;
  }
}

// ---------------------------------------------------------------------------
// Restaurador de dinámica: devuelve ataques (púa) y expresividad que la IA
// aplastó. "attack" realza transitorios, "sustain" la cola de la nota,
// "expand" amplía la micro-dinámica alrededor del nivel medio (descompresión).
// ---------------------------------------------------------------------------
class RestoreProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [kp('enabled', 1, 0, 1), kp('attack', 0, -100, 100), kp('sustain', 0, -100, 100), kp('expand', 0, 0, 100), kp('crisp', 0, 0, 100)];
  }
  constructor() { super(); this.fe = 0; this.se = 0; this.me = 0; this.le = 0; this.gs = 1; this.hf = 0; this.hs = 0; this.gh = 1; this.lp = [0, 0, 0, 0, 0, 0, 0, 0]; }
  process(inputs, outputs, p) {
    const inp = inputs[0], out = outputs[0];
    if (!inp || inp.length === 0) return true;
    if (p.enabled[0] < 0.5) { passThrough(inp, out); return true; }
    const n = inp[0].length, nc = inp.length;
    const att = p.attack[0] / 100, sus = p.sustain[0] / 100, exp = p.expand[0] / 100, crisp = p.crisp[0] / 100;
    const fA = coef(0.3), fR = coef(15), sA = coef(20), sR = coef(120), mA = coef(5), mR = coef(40), lC = coef(400);
    const hA = coef(0.2), hR = coef(8), hsA = coef(15), hsR = coef(80);
    const sm = coef(0.7), smh = coef(0.3);
    const lpa = 1 - Math.exp(-2 * Math.PI * 2500 / sampleRate); // separa la banda del "chasquido" (> 2,5 kHz)
    let fe = this.fe, se = this.se, me = this.me, le = this.le, gs = this.gs, hf = this.hf, hs = this.hs, gh = this.gh;
    const lp = this.lp;
    for (let i = 0; i < n; i++) {
      let a = 0, ah = 0;
      for (let c = 0; c < nc; c++) {
        const x = inp[c][i]; const v = Math.abs(x); if (v > a) a = v;
        lp[c] += lpa * (x - lp[c]); const hv = Math.abs(x - lp[c]); if (hv > ah) ah = hv;
      }
      fe = a > fe ? fA * fe + (1 - fA) * a : fR * fe + (1 - fR) * a;
      se = a > se ? sA * se + (1 - sA) * a : sR * se + (1 - sR) * a;
      me = a > me ? mA * me + (1 - mA) * a : mR * me + (1 - mR) * a;
      le = lC * le + (1 - lC) * me;
      hf = ah > hf ? hA * hf + (1 - hA) * ah : hR * hf + (1 - hR) * ah;
      hs = ah > hs ? hsA * hs + (1 - hsA) * ah : hsR * hs + (1 - hsR) * ah;
      let gdb = 0;
      if (fe > 1e-5) {
        const d = toDb(fe) - toDb(se);
        if (d > 0) gdb += att * Math.min(d, 15) * 0.9;
        else gdb += sus * Math.min(-d, 15) * 0.5;
        if (exp > 0 && le > 1e-5) {
          const e = Math.max(-12, Math.min(12, toDb(me) - toDb(le)));
          gdb += exp * e * 0.6;
        }
      }
      gdb = Math.max(-15, Math.min(15, gdb));
      gs = sm * gs + (1 - sm) * fromDb(gdb);
      let ghdb = 0;
      if (crisp > 0 && hf > 1e-5) { const dh = toDb(hf) - toDb(hs); if (dh > 0) ghdb = crisp * Math.min(dh, 12) * 0.9; }
      gh = smh * gh + (1 - smh) * fromDb(ghdb);
      for (let c = 0; c < out.length; c++) {
        const cc = Math.min(c, nc - 1), x = inp[cc][i];
        out[c][i] = (lp[cc] + (x - lp[cc]) * gh) * gs;
      }
    }
    this.fe = fe; this.se = se; this.me = me; this.le = le; this.gs = gs; this.hf = hf; this.hs = hs; this.gh = gh;
    return true;
  }
}

// ---------------------------------------------------------------------------
// Compresor de rodilla suave, ataque medio y release natural
// ---------------------------------------------------------------------------
class CompProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [kp('enabled', 1, 0, 1), kp('threshold', -18, -80, 0), kp('ratio', 2, 1, 20), kp('attack', 25, 0.1, 300),
      kp('release', 200, 5, 3000), kp('knee', 6, 0, 24), kp('makeup', 0, -24, 24)];
  }
  constructor() { super(); this.grs = 0; this.cnt = 0; this.minGr = 0; }
  process(inputs, outputs, p) {
    const inp = inputs[0], out = outputs[0];
    if (!inp || inp.length === 0) return true;
    if (p.enabled[0] < 0.5) { passThrough(inp, out); this.grs = 0; return true; }
    const n = inp[0].length, nc = inp.length;
    const thr = p.threshold[0], slope = 1 / p.ratio[0] - 1, knee = p.knee[0], mk = p.makeup[0];
    const cA = coef(p.attack[0]), cR = coef(p.release[0]);
    let grs = this.grs, minGr = this.minGr;
    for (let i = 0; i < n; i++) {
      let a = 0;
      for (let c = 0; c < nc; c++) { const v = Math.abs(inp[c][i]); if (v > a) a = v; }
      const over = toDb(a) - thr;
      let gr = 0;
      if (knee > 0 && 2 * Math.abs(over) <= knee) gr = slope * (over + knee / 2) ** 2 / (2 * knee);
      else if (over > 0) gr = slope * over;
      grs = gr < grs ? cA * grs + (1 - cA) * gr : cR * grs + (1 - cR) * gr;
      if (grs < minGr) minGr = grs;
      const g = fromDb(grs + mk);
      for (let c = 0; c < out.length; c++) out[c][i] = inp[Math.min(c, nc - 1)][i] * g;
    }
    this.grs = grs;
    if (++this.cnt >= 16) { this.port.postMessage(minGr); this.cnt = 0; minGr = 0; }
    this.minGr = minGr;
    return true;
  }
}

// ---------------------------------------------------------------------------
// Fase y estéreo: alinear canales, polaridad, graves en mono y anchura
//   align > 0 retrasa L esas muestras (R llegaba tarde); align < 0 retrasa R
// ---------------------------------------------------------------------------
class StereoProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [kp('width', 1, 0, 2), kp('flip', 0, 0, 1), kp('align', 0, -256, 256), kp('monobass', 0, 0, 1), kp('monofreq', 120, 20, 500)];
  }
  constructor() {
    super(); this.N = 512; this.bL = new Float32Array(512); this.bR = new Float32Array(512); this.w = 0;
    this.z1 = 0; this.z2 = 0; this.fc = -1;
  }
  setHP(fc) { // Butterworth paso alto; dos en cascada = Linkwitz-Riley 4º orden sobre el lateral
    const w0 = 2 * Math.PI * fc / sampleRate, al = Math.sin(w0) / (2 * Math.SQRT1_2), cs = Math.cos(w0), a0 = 1 + al;
    this.b0 = (1 + cs) / 2 / a0; this.b1 = -(1 + cs) / a0; this.b2 = this.b0; this.a1 = -2 * cs / a0; this.a2 = (1 - al) / a0; this.fc = fc;
  }
  process(inputs, outputs, p) {
    const inp = inputs[0], out = outputs[0];
    if (!inp || inp.length === 0) return true;
    if (inp.length < 2 || out.length < 2) { passThrough(inp, out); return true; }
    const w = p.width[0], flip = p.flip[0] > 0.5, al = Math.round(p.align[0]), mb = p.monobass[0] > 0.5;
    if (Math.abs(w - 1) < 1e-3 && !flip && al === 0 && !mb) { passThrough(inp, out); return true; }
    if (mb && p.monofreq[0] !== this.fc) this.setHP(p.monofreq[0]);
    const L = inp[0], R = inp[1], N = this.N, dL = al > 0 ? al : 0, dR = al < 0 ? -al : 0;
    let wi = this.w, z1 = this.z1, z2 = this.z2, z3 = this.z3 || 0, z4 = this.z4 || 0;
    for (let i = 0; i < L.length; i++) {
      this.bL[wi] = L[i]; this.bR[wi] = R[i];
      const l = this.bL[(wi - dL + N) % N];
      let r = this.bR[(wi - dR + N) % N];
      wi = (wi + 1) % N;
      if (flip) r = -r;
      const m = (l + r) * 0.5;
      let sd = (l - r) * 0.5;
      if (mb) { // lateral por un paso alto LR4: por debajo de monofreq queda mono puro
        const y = this.b0 * sd + z1; z1 = this.b1 * sd - this.a1 * y + z2; z2 = this.b2 * sd - this.a2 * y;
        const y2 = this.b0 * y + z3; z3 = this.b1 * y - this.a1 * y2 + z4; z4 = this.b2 * y - this.a2 * y2;
        sd = y2;
      }
      sd *= w;
      out[0][i] = m + sd; out[1][i] = m - sd;
    }
    this.w = wi; this.z1 = z1; this.z2 = z2; this.z3 = z3; this.z4 = z4;
    return true;
  }
}

// ---------------------------------------------------------------------------
// Limitador final (techo) + medidores de pico
// ---------------------------------------------------------------------------
class LimiterProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() { return [kp('ceiling', -1, -24, 0), kp('release', 80, 5, 1000)]; }
  constructor() { super(); this.g = 1; this.cnt = 0; this.pk = [0, 0]; this.minG = 1; }
  process(inputs, outputs, p) {
    const inp = inputs[0], out = outputs[0];
    if (!inp || inp.length === 0) return true;
    const n = inp[0].length, nc = inp.length;
    const ceil = fromDb(p.ceiling[0]), r = coef(p.release[0]);
    let g = this.g, minG = this.minG;
    for (let i = 0; i < n; i++) {
      let a = 0;
      for (let c = 0; c < nc; c++) { const v = Math.abs(inp[c][i]); if (v > a) a = v; }
      const t = a * g > ceil ? ceil / a : 1;
      g = t < g ? t : r * g + (1 - r) * Math.min(1, t);
      if (a * g > ceil) g = ceil / a;
      if (g < minG) minG = g;
      for (let c = 0; c < out.length; c++) {
        let y = inp[Math.min(c, nc - 1)][i] * g;
        if (y > ceil) y = ceil; else if (y < -ceil) y = -ceil;
        out[c][i] = y;
        const ay = Math.abs(y);
        if (c < 2 && ay > this.pk[c]) this.pk[c] = ay;
      }
    }
    this.g = g;
    if (++this.cnt >= 12) {
      this.port.postMessage({ l: this.pk[0], r: out.length > 1 ? this.pk[1] : this.pk[0], gr: toDb(minG) });
      this.cnt = 0; this.pk = [0, 0]; minG = 1;
    }
    this.minG = minG;
    return true;
  }
}

// ---------------------------------------------------------------------------
// EQ dinámica supresora de asperezas (2 bandas). Cada banda compara su nivel
// con el de toda la señal y con su propio promedio: solo recorta cuando la
// banda "salta" (picos de deterioro del códec), y vuelve a 0 dB sola.
// ---------------------------------------------------------------------------
function rbj(type, f, q, gdb) {
  const w0 = 2 * Math.PI * Math.min(f, sampleRate * 0.45) / sampleRate, cs = Math.cos(w0), al = Math.sin(w0) / (2 * q);
  let b0, b1, b2, a0, a1, a2;
  if (type === 'bp') { b0 = al; b1 = 0; b2 = -al; a0 = 1 + al; a1 = -2 * cs; a2 = 1 - al; }
  else { const A = Math.pow(10, gdb / 40); b0 = 1 + al * A; b1 = -2 * cs; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * cs; a2 = 1 - al / A; }
  return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
}
class DeHarshProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [kp('enabled', 1, 0, 1), kp('f1', 3200, 1000, 8000), kp('f2', 7000, 3000, 16000), kp('depth', 6, 0, 18), kp('sens', 4, 0, 12), kp('r1', 99, -60, 99), kp('r2', 99, -60, 99)];
  }
  constructor() {
    super();
    this.b = [0, 1].map(() => ({ sc: null, scz: [0, 0], env: 0, full: 0, avg: 0, init: false, gr: 0, co: null, z: [[0, 0], [0, 0]], f: -1 }));
    this.cnt = 0; this.minGr = [0, 0];
  }
  process(inputs, outputs, p) {
    const inp = inputs[0], out = outputs[0];
    if (!inp || inp.length === 0) return true;
    if (p.enabled[0] < 0.5 || p.depth[0] <= 0) { passThrough(inp, out); return true; }
    const n = inp[0].length, nc = inp.length;
    const freqs = [p.f1[0], p.f2[0]], refs = [p.r1[0], p.r2[0]], depth = p.depth[0], sens = p.sens[0];
    const eA = coef(2), eR = coef(60), avgC = coef(2500), gA = coef(10), gR = coef(90); // 10 ms: deja pasar el golpe de púa
    for (let c = 0; c < out.length; c++) out[c].set(inp[Math.min(c, nc - 1)]);
    for (let bi = 0; bi < 2; bi++) {
      const B = this.b[bi];
      if (B.f !== freqs[bi]) { B.sc = rbj('bp', freqs[bi], 1.4, 0); B.f = freqs[bi]; }
      const [s0, s1, s2, s3, s4] = B.sc;
      for (let i = 0; i < n; i++) {
        let m = 0; for (let c = 0; c < nc; c++) m += inp[c][i]; m /= nc;
        const y = s0 * m + B.scz[0]; B.scz[0] = s1 * m - s3 * y + B.scz[1]; B.scz[1] = s2 * m - s4 * y;
        const ay = Math.abs(y), am = Math.abs(m);
        B.env = ay > B.env ? eA * B.env + (1 - eA) * ay : eR * B.env + (1 - eR) * ay;
        B.full = am > B.full ? eA * B.full + (1 - eA) * am : eR * B.full + (1 - eR) * am;
        let target = 0;
        if (B.full > 1e-4) {
          const d = toDb(B.env) - toDb(B.full);
          if (!B.init) { B.avg = d; B.init = true; }
          B.avg = avgC * B.avg + (1 - avgC) * d;
          // compara con su propio promedio y, si se conoce, con el nivel de un espectro equilibrado
          const ex = d - Math.min(B.avg, refs[bi]) - sens;
          if (ex > 0) target = -Math.min(depth, ex * 1.2);
        }
        B.gr = target < B.gr ? gA * B.gr + (1 - gA) * target : gR * B.gr + (1 - gR) * target;
        if ((i & 31) === 0) B.co = rbj('pk', freqs[bi], 1.4, B.gr);
        const [b0, b1, b2, a1, a2] = B.co;
        for (let c = 0; c < out.length; c++) {
          const z = B.z[c] || (B.z[c] = [0, 0]), x = out[c][i];
          const o = b0 * x + z[0]; z[0] = b1 * x - a1 * o + z[1]; z[1] = b2 * x - a2 * o; out[c][i] = o;
        }
        if (B.gr < this.minGr[bi]) this.minGr[bi] = B.gr;
      }
    }
    if (++this.cnt >= 16) { this.port.postMessage(this.minGr.slice()); this.cnt = 0; this.minGr = [0, 0]; }
    return true;
  }
}

registerProcessor('deharsh-processor', DeHarshProcessor);
registerProcessor('gate-processor', GateProcessor);
registerProcessor('restore-processor', RestoreProcessor);
registerProcessor('comp-processor', CompProcessor);
registerProcessor('stereo-processor', StereoProcessor);
registerProcessor('limiter-processor', LimiterProcessor);
