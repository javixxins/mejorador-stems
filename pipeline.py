"""Cadena de mezcla en Python: la misma que la app ejecuta en el navegador.

Recibe las versiones que genera engine.process (x, n, a, na, r, h, e) y los
parámetros (los automáticos o los tuyos) y devuelve el audio final.
Los procesadores dinámicos trabajan por bloques de 32 muestras (<1 ms) con
envolventes de ataque/release: mismo comportamiento que la app, sin latencia.
"""
import numpy as np
from scipy import signal

BLOCK = 32


# ----------------------------------------------------------------------------
# Filtros biquad (mismas fórmulas que los BiquadFilter de WebAudio)
# ----------------------------------------------------------------------------
def biquad(kind, f, sr, q=0.7071, gain_db=0.0):
    f = min(f, sr * 0.475)
    w0 = 2 * np.pi * f / sr
    cs, sn = np.cos(w0), np.sin(w0)
    A = 10 ** (gain_db / 40)
    if kind == "lowpass":
        al = sn / (2 * q)
        b = [(1 - cs) / 2, 1 - cs, (1 - cs) / 2]; a = [1 + al, -2 * cs, 1 - al]
    elif kind == "highpass":
        al = sn / (2 * q)
        b = [(1 + cs) / 2, -(1 + cs), (1 + cs) / 2]; a = [1 + al, -2 * cs, 1 - al]
    elif kind == "bandpass":
        al = sn / (2 * q)
        b = [al, 0, -al]; a = [1 + al, -2 * cs, 1 - al]
    elif kind == "peaking":
        al = sn / (2 * q)
        b = [1 + al * A, -2 * cs, 1 - al * A]; a = [1 + al / A, -2 * cs, 1 - al / A]
    elif kind in ("lowshelf", "highshelf"):
        al = sn / 2 * np.sqrt(2)  # pendiente S = 1
        sA = 2 * np.sqrt(A) * al
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * cs + sA), 2 * A * ((A - 1) - (A + 1) * cs), A * ((A + 1) - (A - 1) * cs - sA)]
            a = [(A + 1) + (A - 1) * cs + sA, -2 * ((A - 1) + (A + 1) * cs), (A + 1) + (A - 1) * cs - sA]
        else:
            b = [A * ((A + 1) + (A - 1) * cs + sA), -2 * A * ((A - 1) + (A + 1) * cs), A * ((A + 1) + (A - 1) * cs - sA)]
            a = [(A + 1) - (A - 1) * cs + sA, 2 * ((A - 1) - (A + 1) * cs), (A + 1) - (A - 1) * cs - sA]
    else:
        raise ValueError(kind)
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return b, a


def filt(y, kind, f, sr, q=0.7071, gain_db=0.0):
    if kind in ("peaking", "lowshelf", "highshelf") and abs(gain_db) < 1e-3:
        return y
    b, a = biquad(kind, f, sr, q, gain_db)
    return signal.lfilter(b, a, y, axis=-1)


# ----------------------------------------------------------------------------
# Envolventes por bloques
# ----------------------------------------------------------------------------
def coef(ms, sr, block=BLOCK):
    return np.exp(-block / max(1.0, ms / 1000 * sr))


def block_peak(y):
    """Pico por bloque (enlazado entre canales). y: (canales, n)."""
    n = y.shape[1]
    nb = int(np.ceil(n / BLOCK))
    pad = np.zeros((y.shape[0], nb * BLOCK - n))
    v = np.abs(np.concatenate([y, pad], axis=1)).reshape(y.shape[0], nb, BLOCK).max(axis=2)
    return v.max(axis=0)


def follow(v, att, rel, start=0.0):
    out = np.empty_like(v)
    e = start
    for i, x in enumerate(v):
        e = att * e + (1 - att) * x if x > e else rel * e + (1 - rel) * x
        out[i] = e
    return out


def to_samples(g, n):
    """Ganancia por bloque -> por muestra (interpolación lineal, sin escalones)."""
    xb = (np.arange(len(g)) + 0.5) * BLOCK
    return np.interp(np.arange(n), xb, g)


def dbv(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


# ----------------------------------------------------------------------------
# Procesadores
# ----------------------------------------------------------------------------
def gate(y, sr, P):
    if not P.get("gate.on"):
        return y
    env = follow(block_peak(y), coef(0.5, sr), coef(30, sr))
    lvl = dbv(env)
    thr, ratio, rng = P["gate.thr"], P["gate.ratio"], P["gate.range"]
    hold = int(P["gate.hold"] / 1000 * sr / BLOCK)
    tgt = np.ones_like(lvl)
    hc = 0
    for i, l in enumerate(lvl):
        if l >= thr:
            hc = hold
        elif hc > 0:
            hc -= 1
        else:
            tgt[i] = 10 ** (max(-(thr - l) * (ratio - 1), -rng) / 20)
    g = follow(tgt, coef(P["gate.att"], sr), coef(P["gate.rel"], sr), 1.0)
    return y * to_samples(g, y.shape[1])


def eq(y, sr, P):
    if not P.get("eq.on"):
        return y
    hp = P["hpf.freq"]
    y = filt(y, "highpass", hp, sr)
    if P.get("hpf.slope", 12) == 24:
        y = filt(y, "highpass", hp, sr)
    t = P.get("eq.tilt", 0)
    y = filt(y, "lowshelf", 1000, sr, gain_db=-t)
    y = filt(y, "highshelf", 1000, sr, gain_db=t)
    y = filt(y, "lowshelf", P["eq.low_f"], sr, gain_db=P["eq.low_g"])
    y = filt(y, "peaking", P["eq.mud_f"], sr, 1.0, P["eq.mud_g"])
    y = filt(y, "peaking", P["eq.mid_f"], sr, 0.9, P["eq.mid_g"])
    y = filt(y, "peaking", P["eq.pres_f"], sr, 1.0, P["eq.pres_g"])
    y = filt(y, "peaking", P["eq.harsh_f"], sr, 1.4, P["eq.harsh_g"])
    y = filt(y, "highshelf", P["eq.air_f"], sr, gain_db=P["eq.air_g"])
    if P["eq.lpf"] < 19900:
        y = filt(filt(y, "lowpass", P["eq.lpf"], sr), "lowpass", P["eq.lpf"], sr)
    for n_ in P.get("notches", []):
        if n_.get("on"):
            y = filt(y, "peaking", n_["f"], sr, max(0.3, n_["q"]), n_["g"])
    return y


def deharsh(y, sr, P):
    """EQ dinámica de 2 bandas (misma lógica que la app)."""
    if not P.get("deh.on") or P.get("deh.depth", 0) <= 0:
        return y
    mono = y.mean(axis=0)
    depth, sens = P["deh.depth"], P["deh.sens"]
    for f, ref in ((P["deh.f1"], P.get("deh.r1", 99)), (P["deh.f2"], P.get("deh.r2", 99))):
        b, a = biquad("bandpass", f, sr, 1.4)
        sc = signal.lfilter(b, a, mono)
        eb = follow(block_peak(sc[None]), coef(2, sr), coef(60, sr))
        ef = follow(block_peak(mono[None]), coef(2, sr), coef(60, sr))
        d = dbv(eb) - dbv(ef)
        alive = ef > 1e-4
        avg = np.empty_like(d); ac = coef(2500, sr); m = d[alive][0] if alive.any() else 0.0
        for i in range(len(d)):
            if alive[i]:
                m = ac * m + (1 - ac) * d[i]
            avg[i] = m
        ex = d - np.minimum(avg, ref) - sens
        tgt = np.where(alive & (ex > 0), -np.minimum(depth, ex * 1.2), 0.0)
        gr = follow(-tgt, coef(10, sr), coef(90, sr))  # follow trabaja con positivos
        gr = -gr
        # filtro peaking variable: se recalcula cada 256 muestras con estado continuo
        out = np.empty_like(y)
        step = 256
        zi = np.zeros((y.shape[0], 2))
        for s0 in range(0, y.shape[1], step):
            g = float(gr[min(len(gr) - 1, s0 // BLOCK)])
            bb, aa = biquad("peaking", f, sr, 1.4, g)
            out[:, s0:s0 + step], zi = signal.lfilter(bb, aa, y[:, s0:s0 + step], axis=-1, zi=zi)
        y = out
    return y


def restore_dyn(y, sr, P):
    if not P.get("dyn.on"):
        return y
    att, sus, exp_, crisp = P["dyn.attack"] / 100, P["dyn.sustain"] / 100, P["dyn.expand"] / 100, P.get("dyn.crisp", 0) / 100
    pk = block_peak(y)
    fe = follow(pk, coef(0.3, sr), coef(15, sr)); se = follow(pk, coef(20, sr), coef(120, sr))
    me = follow(pk, coef(5, sr), coef(40, sr))
    le = signal.lfilter([1 - coef(400, sr)], [1, -coef(400, sr)], me)
    d = dbv(fe) - dbv(se)
    gdb = np.where(d > 0, att * np.minimum(d, 15) * 0.9, sus * np.minimum(-d, 15) * 0.5)
    if exp_ > 0:
        gdb += exp_ * np.clip(dbv(me) - dbv(le), -12, 12) * 0.6 * (le > 1e-5)
    gdb = np.clip(np.where(fe > 1e-5, gdb, 0), -15, 15)
    g = to_samples(10 ** (gdb / 20), y.shape[1])
    if crisp > 0:
        lpa = 1 - np.exp(-2 * np.pi * 2500 / sr)
        lp = signal.lfilter([lpa], [1, -(1 - lpa)], y, axis=-1)
        hi = y - lp
        hpk = block_peak(hi)
        hf = follow(hpk, coef(0.2, sr), coef(8, sr)); hs = follow(hpk, coef(15, sr), coef(80, sr))
        dh = dbv(hf) - dbv(hs)
        gh = 10 ** (np.where((dh > 0) & (hf > 1e-5), crisp * np.minimum(dh, 12) * 0.9, 0) / 20)
        y = lp + hi * to_samples(gh, y.shape[1])
    return y * g


def compressor(y, sr, P):
    if not P.get("comp.on"):
        return y
    thr, ratio, knee, mk = P["comp.thr"], P["comp.ratio"], P["comp.knee"], P["comp.makeup"]
    lvl = dbv(block_peak(y))
    over = lvl - thr
    slope = 1 / ratio - 1
    gr = np.where(over > knee / 2, slope * over, 0.0)
    if knee > 0:
        k = np.abs(2 * over) <= knee
        gr = np.where(k, slope * (over + knee / 2) ** 2 / (2 * knee), gr)
    grs = -follow(-gr, coef(P["comp.att"], sr), coef(P["comp.rel"], sr))
    return y * to_samples(10 ** ((grs + mk) / 20), y.shape[1])


def mic_character(y, sr, P):
    if not P.get("mic.on"):
        return y
    a = P["mic.amount"] / 100
    y = filt(y, "lowshelf", 130, sr, gain_db=1.5 * a)
    y = filt(y, "peaking", 220, sr, 0.8, 0.6 * a)
    y = filt(y, "peaking", 4800, sr, 0.7, 1.8 * a)
    y = filt(y, "highshelf", 11000, sr, gain_db=2.5 * a)
    ex = P["mic.excite"] / 100 * 0.35
    if ex > 0:
        e = filt(filt(y, "highpass", 3000, sr), "lowpass", 8000, sr)
        e = np.tanh(2.5 * np.clip(e * 4, -1, 1))
        y = y + filt(e, "highpass", 5000, sr) * ex
    s = P["mic.sat"] / 100
    if s >= 0.005:
        d, b = 0.3 + 2.2 * s, 0.25 * s
        tb = np.tanh(d * b)
        xx = np.clip(y, -4, 4)
        y = (np.tanh(d * (xx + b)) - tb) / (d * (1 - tb * tb))
    return filt(y, "highpass", 10, sr)


def stereo(y, sr, P):
    if y.shape[0] < 2:
        return y
    L, R = y[0].copy(), y[1].copy()
    al = int(P.get("stereo.align", 0))
    if al > 0:
        L = np.concatenate([np.zeros(al), L[:-al]])
    elif al < 0:
        R = np.concatenate([np.zeros(-al), R[:al]])
    if P.get("stereo.flip"):
        R = -R
    m, sd = (L + R) / 2, (L - R) / 2
    if P.get("stereo.monobass"):
        # Linkwitz-Riley 4º orden (2 Butterworth en cascada) solo sobre el lateral:
        # por debajo del corte queda mono puro y el centro no se toca
        b, a = biquad("highpass", P.get("stereo.monofreq", 120), sr)
        sd = signal.lfilter(b, a, signal.lfilter(b, a, sd))
    sd = sd * P["stereo.width"] / 100
    return np.stack([m + sd, m - sd])


def make_ir(sr, kind, size, pre, tone):
    n = int(sr * (pre + size * 1.3)) + 1
    rng = np.random.default_rng(12345)
    p0 = int(pre * sr)
    t = np.arange(n - p0) / sr
    cut0 = 2000 + tone * 14000
    damp = 1.2 + (1 - tone) * 3
    ir = np.zeros((2, n))
    for ch in range(2):
        noise = rng.uniform(-1, 1, n - p0)
        fc = np.maximum(400, cut0 * np.exp(-t * damp / size))
        a = 1 - np.exp(-2 * np.pi * fc / sr)
        lp = np.empty_like(noise); z = 0.0
        for i in range(len(noise)):  # filtro de amortiguación variable
            z += a[i] * (noise[i] - z); lp[i] = z
        fade = 1 if kind == "placa" else np.minimum(1, t / 0.008)
        ir[ch, p0:] = lp * np.exp(-6.9 * t / size) * fade
    ir /= np.sqrt(np.sum(ir ** 2) / 2) + 1e-12
    return ir


def reverb(y, sr, P):
    if not P.get("rev.on") or P["rev.mix"] <= 0:
        return y
    ir = make_ir(sr, P["rev.type"], P["rev.size"], P["rev.predelay"] / 1000, P["rev.tone"] / 100)
    src = filt(y, "highpass", 200, sr)
    if src.shape[0] == 1:
        src = np.repeat(src, 2, axis=0); y = np.repeat(y, 2, axis=0)
    wet = np.stack([signal.fftconvolve(src[c], ir[c])[: y.shape[1]] for c in range(2)])
    return y + wet * P["rev.mix"] / 100 * 0.9 * 0.12


def limiter(y, sr, P):
    ceil = 10 ** (P["out.ceil"] / 20)
    pk = block_peak(y)
    need = np.minimum(1, ceil / np.maximum(pk, 1e-12))
    r = coef(80, sr)
    g = np.empty_like(need); cur = 1.0
    for i, t in enumerate(need):
        cur = t if t < cur else r * cur + (1 - r) * t
        g[i] = cur
    # el mínimo del bloque y del anterior evita que la interpolación deje pasar picos
    g = np.minimum(g, np.concatenate([[1.0], g[:-1]]))
    y = y * to_samples(g, y.shape[1])
    return np.clip(y, -ceil, ceil)


def true_peak(y, sr):
    up = signal.resample_poly(y, 4, 1, axis=-1)
    return float(np.max(np.abs(up)))


# ----------------------------------------------------------------------------
# Cadena completa
# ----------------------------------------------------------------------------
def mix_buffers(B, P):
    pn, pa = P["clean.nr"] / 100, P["clean.art"] / 100
    y = ((1 - pn) * (1 - pa) * B["x"] + pn * (1 - pa) * B["n"] + (1 - pn) * pa * B["a"] + pn * pa * B["na"])
    y = y + P.get("rep.fill", 0) / 100 * B["r"] + P.get("rep.hf", 0) / 100 * B["h"]
    if "e" in B:
        y = y + P.get("rep.bal", 0) / 100 * B["e"]
    return y


def render(B, P, sr, progress=lambda f, s: None):
    """B: dict de arrays (canales, muestras). Devuelve el audio final (canales, muestras)."""
    y = mix_buffers(B, P).astype(np.float64)
    steps = [("Puerta de ruido suave", gate), ("Ecualización correctiva", eq), ("EQ dinámica antiasperezas", deharsh),
             ("Restaurando dinámica y ataques", restore_dyn), ("Compresor", compressor),
             ("Carácter de micro de estudio", mic_character), ("Fase y estéreo", stereo), ("Reverb", reverb)]
    for i, (name, fn) in enumerate(steps):
        progress(i / (len(steps) + 2), name)
        y = fn(y, sr, P)
    y = y * 10 ** (P.get("out.gain", 0) / 20)
    level = P.get("out.level", "tp")
    if level != "tp":  # con normalización por pico verdadero no hace falta limitar (y no se recorta nada)
        progress(len(steps) / (len(steps) + 2), "Limitador")
        y = limiter(y, sr, P)
    progress((len(steps) + 1) / (len(steps) + 2), "Normalizando nivel")
    if level == "tp":
        tp = true_peak(y, sr)
        if tp > 1e-6:
            y = y * (10 ** (-1.0 / 20) / tp)
    elif level == "match":
        g = np.sqrt(np.mean(B["x"] ** 2) / (np.mean(y ** 2) + 1e-20))
        y = y * np.clip(g, 0.25, 4)
        y = limiter(y, sr, P)
    return y.astype(np.float32)
