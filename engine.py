"""Motor de análisis y limpieza de stems.

Hace en el servidor la parte pesada que no se puede hacer en tiempo real:
  * decodificación (wav/mp3/…), quitar DC y reparar clipping
  * análisis (ruido de fondo, compresión, resonancias, balance, artefactos)
  * limpieza espectral en dos máscaras independientes:
        Mn = reducción de ruido (puerta espectral suave que respeta armónicos)
        Ma = eliminación de artefactos IA ("chisporroteo"/"agua" en agudos)
Se guardan cuatro versiones: X, X·Mn, X·Ma, X·Mn·Ma. Como la cantidad de
reducción es lineal en la máscara, el navegador puede mezclarlas con pesos
bilineales y obtener EXACTAMENTE cualquier % de ruido/artefactos en tiempo real.
"""
import json
import os
import subprocess

import numpy as np
import soundfile as sf

import restore
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter, minimum_filter1d, uniform_filter, uniform_filter1d

EPS = 1e-10

SLOPE_TARGET = {"guitarra_limpia": -5.0, "guitarra_distorsion": -4.5, "guitarra_acustica": -4.5, "bajo": -7.5,
                "voz": -5.0, "bateria": -3.5, "teclados": -4.5, "otro": -4.5}

PROFILES = {
    # hpf, lpf, objetivos de balance por octava relativos a 1 kHz
    "guitarra_limpia": dict(hpf=70, target={125: -2, 250: 1, 500: 1, 2000: -2, 4000: -6, 8000: -12}),
    "guitarra_distorsion": dict(hpf=80, target={125: -3, 250: 0, 500: 1, 2000: 0, 4000: -4, 8000: -14}),
    "guitarra_acustica": dict(hpf=70, target={125: -3, 250: 0, 500: 0, 2000: -2, 4000: -5, 8000: -10}),
    "bajo": dict(hpf=30, target={125: 8, 250: 5, 500: 2, 2000: -5, 4000: -12, 8000: -20}),
    "voz": dict(hpf=80, target={125: -6, 250: -1, 500: 1, 2000: -1, 4000: -4, 8000: -10}),
    "bateria": dict(hpf=30, target={125: 4, 250: 1, 500: -1, 2000: -2, 4000: -3, 8000: -6}),
    "teclados": dict(hpf=40, target={125: 0, 250: 1, 500: 1, 2000: -2, 4000: -6, 8000: -12}),
    "otro": dict(hpf=30, target={125: 0, 250: 0, 500: 0, 2000: -2, 4000: -5, 8000: -10}),
}


def _interp(x, xs, ys):
    return float(np.interp(x, xs, ys))


def db(x):
    return 20 * np.log10(np.maximum(x, EPS))


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
def load_audio(path, workdir):
    data = None
    if path.lower().endswith((".wav", ".wave", ".aif", ".aiff", ".flac")):
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            data = None
    if data is None:  # mp3 y demás: ffmpeg (decodifica sin errores y respeta el gapless)
        tmp = os.path.join(workdir, "decoded.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-acodec", "pcm_f32le", tmp],
            check=True,
        )
        data, sr = sf.read(tmp, dtype="float32", always_2d=True)
        os.remove(tmp)
    if data.shape[1] > 2:  # stems multicanal raros -> estéreo
        data = data[:, :2]
    return data.T.copy(), int(sr)  # (canales, muestras)


def remove_dc(x, sr):
    b, a = signal.butter(1, 5.0 / (sr / 2), btype="highpass")
    return signal.filtfilt(b, a, x, axis=1).astype(np.float32)


def declip(x):
    """Reconstruye tramos recortados (planos) con interpolación cúbica."""
    peak = float(np.max(np.abs(x)))
    if peak < 0.98:
        return x, 0
    thr = peak * 0.997
    fixed = 0
    for c in range(x.shape[0]):
        ch = x[c]
        flat = np.abs(np.diff(ch)) < 1e-4
        clipped = np.zeros(len(ch), bool)
        clipped[1:] = (np.abs(ch[1:]) >= thr) & flat
        clipped[:-1] |= clipped[1:]
        if clipped.sum() < 3:
            continue
        idx = np.flatnonzero(clipped)
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([idx[0]], idx[breaks + 1]))
        ends = np.concatenate((idx[breaks], [idx[-1]]))
        for s, e in list(zip(starts, ends))[:20000]:
            if e - s < 2 or e - s > 120:
                continue
            l0, r1 = max(0, s - 6), min(len(ch), e + 7)
            xs = np.concatenate((np.arange(l0, s), np.arange(e + 1, r1)))
            if len(xs) < 6:
                continue
            seg = np.arange(s, e + 1)
            y = CubicSpline(xs, ch[xs])(seg)
            lim = thr * 1.6
            ch[seg] = np.clip(y, -lim, lim)
            fixed += 1
    return x, fixed


# ----------------------------------------------------------------------------
# Análisis
# ----------------------------------------------------------------------------
def frame_levels(mono, sr, win_s):
    n = max(1, int(sr * win_s))
    k = len(mono) // n
    if k == 0:
        return np.array([db(np.sqrt(np.mean(mono ** 2)))])
    fr = mono[: k * n].reshape(k, n)
    return db(np.sqrt(np.mean(fr ** 2, axis=1)))


def band_db(f, psd, lo, hi):
    m = (f >= lo) & (f < hi)
    if not m.any():
        return -120.0
    return float(10 * np.log10(np.sum(psd[m]) + 1e-20))


def analyze(x, sr):
    mono = x.mean(axis=0)
    peak_db = float(db(np.max(np.abs(x))))
    rms_db = float(db(np.sqrt(np.mean(mono ** 2))))

    noise_db = max(-110.0, noise_level_db(mono, sr))
    lv400 = frame_levels(mono, sr, 0.4)
    act400 = lv400[lv400 > noise_db + 10]
    if len(act400) > 3:
        dyn_range = float(np.percentile(act400, 95) - np.percentile(act400, 10))
        loud95 = float(np.percentile(act400, 95))
    else:
        dyn_range, loud95 = 0.0, rms_db
    active_rms = float(db(np.sqrt(np.mean(10 ** (act400 / 10))))) if len(act400) else rms_db
    plr = peak_db - active_rms  # relación pico/sonoridad

    f, psd = signal.welch(x, sr, nperseg=8192, axis=-1)
    psd = psd.mean(axis=0)  # media por canal: no se cancela si hay problemas de fase
    ref = band_db(f, psd, 707, 1414)
    octs = {}
    for fc in (63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000):
        octs[fc] = band_db(f, psd, fc / 1.414, fc * 1.414) - ref

    # ancho de banda real (muchos stems de IA cortan en 15-16 kHz)
    psd_db = 10 * np.log10(psd + 1e-20)
    smooth = uniform_filter1d(psd_db, 15)
    top = np.max(smooth[(f > 200) & (f < 5000)])
    above = np.flatnonzero((smooth > top - 60) & (f > 2000))
    bandwidth = float(f[above[-1]]) if len(above) else sr / 2
    # comprobación instante a instante: % de tiempo con contenido real por encima del corte
    fz, _, zz = signal.stft(mono, sr, nperseg=2048, noverlap=1024)
    A = 20 * np.log10(np.abs(zz) + 1e-9)
    act = A.max(axis=0) > A.max() - 50
    hi_band = (fz > bandwidth * 1.08) & (fz < min(sr / 2 * 0.95, bandwidth * 1.08 + 4000))
    if hi_band.any() and act.any():
        ref_lvl = np.percentile(A[(fz > 1000) & (fz < 4000)][:, act], 50, axis=0)
        over = (A[hi_band][:, act].max(axis=0) > ref_lvl - 45)
        hf_beyond = float(100 * over.mean())
    else:
        hf_beyond = 0.0

    # resonancias: picos estrechos del espectro medio que ADEMÁS están presentes
    # casi todo el tiempo (las notas tocadas van y vienen; una resonancia no)
    grid = 200 * 2 ** (np.arange(0, int(24 * np.log2(8000 / 200))) / 24)
    g_db = np.interp(grid, f, psd_db)
    resid = g_db - median_filter(g_db, size=13, mode="nearest")
    peaks, props = signal.find_peaks(resid, height=4.0, width=1, distance=6)
    res = []
    if len(peaks):
        fz, _, zz = signal.stft(mono, sr, nperseg=8192, noverlap=4096)
        sp = db(np.abs(zz))
        loud = sp.max(axis=0) > sp.max() - 50
        sp = sp[:, loud] if loud.sum() > 5 else sp
        gsp = np.array([np.interp(grid, fz, sp[:, i]) for i in range(sp.shape[1])]).T
        gres = gsp - median_filter(gsp, size=(13, 1), mode="nearest")
        cand = []
        for p, h, w in zip(peaks, props["peak_heights"], props["widths"]):
            persist = float(np.mean(gres[p] > 3.0))
            if persist >= 0.75:
                cand.append((p, h, w, persist))
        for p, h, w, persist in sorted(cand, key=lambda t: -t[1])[:4]:
            oct_bw = max(w / 24, 1 / 12)
            q = float(np.clip(np.sqrt(2 ** oct_bw) / (2 ** oct_bw - 1), 2, 8))
            res.append(dict(f=round(float(grid[p])), g=round(-min(4.0, h * 0.4), 1), q=round(q, 1),
                            excess=round(float(h), 1), persist=round(persist, 2)))

    # balance tonal: pendiente del espectro (dB/octava) entre 60 Hz y 12 kHz
    m = (f >= 60) & (f <= min(12000, sr / 2 * 0.9))
    lf = np.log2(f[m])
    pts = np.arange(lf.min(), lf.max(), 1 / 6)
    vals = np.interp(pts, lf, uniform_filter1d(psd_db[m], 9))
    slope = float(np.polyfit(pts, vals, 1)[0])

    # fase / estéreo
    corr, corr_low, lag, lag_gain, polarity = 1.0, 1.0, 0, 0.0, 1
    if x.shape[0] == 2 and np.std(x[0]) > 0 and np.std(x[1]) > 0:
        L, R = x[0].astype(np.float64), x[1].astype(np.float64)
        corr = float(np.corrcoef(L, R)[0, 1])
        b, a = signal.butter(4, 150 / (sr / 2), "lowpass")
        lo = signal.filtfilt(b, a, np.vstack((L, R)), axis=1)
        if np.std(lo[0]) > 1e-6 and np.std(lo[1]) > 1e-6:
            corr_low = float(np.corrcoef(lo[0], lo[1])[0, 1])
        # desfase temporal entre canales (hasta ±1,5 ms)
        b, a = signal.butter(4, 1500 / (sr / 2), "lowpass")
        seg = slice(0, min(len(L), sr * 60))
        l2, r2 = signal.filtfilt(b, a, np.vstack((L[seg], R[seg])), axis=1)
        maxlag = int(sr * 0.0015)
        cc = signal.correlate(r2, l2, mode="full", method="fft")
        mid = len(l2) - 1
        win = cc[mid - maxlag: mid + maxlag + 1] / (np.sqrt(np.sum(l2 ** 2) * np.sum(r2 ** 2)) + 1e-12)
        k = int(np.argmax(np.abs(win)))
        best, at0 = float(abs(win[k])), float(abs(win[maxlag]))
        if k != maxlag and best - at0 > 0.08:
            lag, lag_gain = k - maxlag, best - at0
        else:
            k = maxlag
        if win[k] < -0.5:  # canales en contrafase (ya alineados)
            polarity = -1

    return dict(
        peak_db=round(peak_db, 1), rms_db=round(rms_db, 1), noise_db=round(noise_db, 1),
        dyn_range=round(dyn_range, 1), plr=round(plr, 1), loud95=round(loud95, 1),
        octaves={str(k): round(v, 1) for k, v in octs.items()},
        bandwidth=round(bandwidth), hf_beyond_pct=round(hf_beyond, 1), resonances=res, corr=round(corr, 2), corr_low=round(corr_low, 2),
        lag=int(lag), lag_ms=round(lag / sr * 1000, 2), lag_gain=round(lag_gain, 2), polarity=polarity, slope=round(slope, 2),
    )


# ----------------------------------------------------------------------------
# Limpieza espectral
# ----------------------------------------------------------------------------
NFFT, HOP = 2048, 512


def spectral_masks(z, sr, noise_prof):
    """Devuelve (Mn, Ma, art_db) para un canal. z = STFT compleja (bins x frames)."""
    mag = np.abs(z).astype(np.float32)
    nb, nf = mag.shape
    freqs = np.linspace(0, sr / 2, nb)

    # --- Mn: puerta espectral suave con perfil de ruido ADAPTATIVO ------------
    # El perfil global sirve para el siseo constante. En agudos (>1,5 kHz) además se
    # sigue el suelo de ruido local (estadística de mínimos en ventanas de 2 s): así
    # se elimina el "hash" digital que aparece solo en algunas partes de la canción.
    mag_s = uniform_filter1d(mag, 3, axis=1)
    win = max(9, int(2.0 * sr / HOP))
    local = minimum_filter1d(uniform_filter1d(mag, 5, axis=1), win, axis=1, mode="nearest") * 1.5
    local = median_filter(local, size=(9, 1), mode="nearest")
    w_hf = np.clip((freqs - 1000) / 1000, 0, 1)[:, None].astype(np.float32)
    prof_tf = np.maximum(noise_prof[:, None], local * w_hf)
    del local
    snr = db(mag_s) - db(prof_tf)
    del prof_tf
    del mag_s
    mn = 1.0 / (1.0 + np.exp(-(snr - 6.0) / 2.5))
    del snr
    mn = uniform_filter(mn, size=(3, 5)).astype(np.float32)
    floor = 10 ** (-24 / 20)
    mn = floor + (1 - floor) * mn

    # --- Ma: artefactos IA (destellos inestables en agudos) -------------------
    ma = np.ones_like(mag)
    b_lo = int(np.searchsorted(freqs, 3000))
    if b_lo < nb - 4:
        hf = mag[b_lo:]
        med = median_filter(hf, size=(1, 9), mode="nearest")
        target = np.minimum(hf, 1.6 * med)
        m = target / (hf + EPS)
        del med, target
        m = uniform_filter(m, size=(3, 1))
        w = np.clip((freqs[b_lo:] - 3000) / 3000, 0, 1)[:, None]
        m = 1 - w * (1 - m)
        # proteger ataques (púa, golpes): no tocar frames de onset
        logm = np.log(mag + 1e-6)
        flux = np.mean(np.maximum(0, np.diff(logm, axis=1)), axis=0)
        del logm
        flux = np.concatenate(([0], flux))
        medf = np.median(flux)
        mad = np.median(np.abs(flux - medf)) + 1e-9
        onset = flux > medf + 4 * mad
        prot = onset.copy()
        for s in (-1, 1, 2):
            prot |= np.roll(onset, s)
        m[:, prot] = 1 - (1 - m[:, prot]) * 0.25
        ma[b_lo:] = m
        hf_e = np.sum(hf ** 2)
        art_db = float(10 * np.log10((hf_e + EPS) / (np.sum((hf * m) ** 2) + EPS)))
    else:
        art_db = 0.0
    return mn, ma.astype(np.float32), art_db


def noise_profile(z):
    mag = np.abs(z)
    e = np.sum(mag ** 2, axis=0)
    live = e > 1e-10
    if live.sum() < 10:
        return np.full(mag.shape[0], 1e-6, np.float32)
    ml = uniform_filter1d(mag[:, live], 5, axis=1)  # suaviza 5 frames
    thr = np.percentile(e[live], 5)
    quiet = ml[:, e[live] <= thr].mean(axis=1)
    p5 = np.percentile(ml, 5, axis=1)
    prof = np.minimum(quiet, p5)
    # quitar picos tonales del perfil: el ruido es espectralmente liso
    prof = np.minimum(prof, median_filter(prof, size=15, mode="nearest"))
    return prof.astype(np.float32) + 1e-9


def noise_level_db(x_mono, sr):
    """Nivel de ruido de fondo (dBFS RMS) a partir del perfil espectral."""
    _, _, z = signal.stft(x_mono, sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)
    prof = noise_profile(z)
    p = np.abs(z) ** 2
    frame_pow = np.mean(np.sum(p, axis=0))
    k = np.mean(x_mono ** 2) / (frame_pow + 1e-20)
    return float(10 * np.log10(k * np.sum(prof.astype(np.float64) ** 2) + 1e-20))


# ----------------------------------------------------------------------------
# Equilibrio adaptativo: tramos mucho más estridentes que las partes sanas
# ----------------------------------------------------------------------------
def adaptive_balance(zc, sr):
    """Devuelve (E aditivo, curva de ganancia dB por frame). Solo REDUCE agudos (>2 kHz,
    en forma de shelf) en los tramos cuya relación agudos/medios supera en más de 6 dB a
    la de las partes sanas de la canción. Variación lenta (~1 s): no toca los ataques."""
    nb, nf = zc.shape
    freqs = np.linspace(0, sr / 2, nb)
    P = np.abs(zc) ** 2
    hf = 10 * np.log10(P[(freqs >= 4000) & (freqs < 16000)].sum(axis=0) + 1e-12)
    md = 10 * np.log10(P[(freqs >= 300) & (freqs < 2000)].sum(axis=0) + 1e-12)
    k = max(3, int(1.0 * sr / HOP))
    hf_s = 10 * np.log10(uniform_filter1d(10 ** (hf / 10), k) + 1e-12)
    md_s = 10 * np.log10(uniform_filter1d(10 ** (md / 10), k) + 1e-12)
    r = hf_s - md_s
    act = md_s > md_s.max() - 40
    if act.sum() < 10:
        return np.zeros((nb, nf), np.float32), np.zeros(nf, np.float32)
    ref = np.percentile(r[act], 25)
    gdb = -np.clip(r - ref - 6, 0, 12) * act
    gdb = uniform_filter1d(gdb, k)
    shape = np.clip((freqs - 2000) / 3000, 0, 1)[:, None]
    G = 10 ** (gdb[None, :] * shape / 20)
    return (G - 1).astype(np.float32), gdb.astype(np.float32)


# ----------------------------------------------------------------------------
# Acumulación de energía en medios-agudos (zona de estridencia)
# ----------------------------------------------------------------------------
def _rbj_bp(f, q, sr):
    w0 = 2 * np.pi * f / sr
    al = np.sin(w0) / (2 * q)
    b = np.array([al, 0, -al]); a = np.array([1 + al, -2 * np.cos(w0), 1 - al])
    return b / a[0], a / a[0]


def buildup(y, sr, slope_target, cutoff=20000):
    """Exceso (dB) de cada tercio de octava 1,6-8 kHz sobre la línea de un espectro
    equilibrado anclada en 160-1600 Hz. Devuelve lista [(f, exceso)] ordenada."""
    f, psd = signal.welch(y, sr, nperseg=8192)
    cs = 160 * 2 ** (np.arange(0, 18) / 3)  # 160 Hz .. ~8 kHz
    cs = cs[cs * 2 ** (1 / 6) < 0.9 * cutoff]  # solo donde hay contenido real
    pts = []
    for c in cs:
        m = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        if m.any():
            pts.append((c, 10 * np.log10(np.mean(psd[m]) + 1e-20)))
    pts = np.array(pts)
    # la acumulación es una "joroba" sobre la tendencia del PROPIO sonido (el brillo
    # general ya lo corrige la inclinación tonal): recta robusta ignorando lo que sobresale
    lx, ly = np.log2(pts[:, 0]), pts[:, 1]
    keep = np.ones(len(lx), bool)
    for _ in range(4):
        k_, c_ = np.polyfit(lx[keep], ly[keep], 1)
        res = ly - (k_ * lx + c_)
        keep = res < np.percentile(res, 70)
    ex = ly - (k_ * lx + c_)
    out = [(float(c), float(e)) for c, e in zip(pts[:, 0], ex) if c >= 1600]
    return sorted(out, key=lambda t: -t[1])


BUILDUP_THR = 4.5  # por debajo, es la forma natural del instrumento


def pick_buildup(bu):
    """De la lista de excesos elige (activo, f1, exceso1, f2, exceso2) en la zona 2-8 kHz."""
    z = [(f, e) for f, e in bu if f >= 2000]
    if not z:
        return False, 3200.0, 0.0, 7000.0, 0.0
    f1, e1 = z[0]
    on = e1 > BUILDUP_THR
    f2, e2 = next(((f, e) for f, e in z[1:] if f >= 5000 and abs(np.log2(f / f1)) >= 0.5), (7000.0, 0.0))
    if not on:
        return False, 3200.0, e1, 7000.0, 0.0
    if e2 <= BUILDUP_THR:
        f2, e2 = 7000.0, 0.0
    return True, float(np.clip(f1, 2000, 6000)), e1, f2, e2


def deharsh_ref(y, sr, f, excess):
    """Nivel 'equilibrado' (banda menos señal completa, dB) que usa la EQ dinámica."""
    b, a = _rbj_bp(f, 1.4, sr)
    bp = signal.lfilter(b, a, y)
    n = int(sr * 0.05)
    k = len(y) // n
    if k < 4:
        return 99.0
    eb = np.sqrt(np.mean(bp[: k * n].reshape(k, n) ** 2, axis=1))
    ef = np.sqrt(np.mean(y[: k * n].reshape(k, n) ** 2, axis=1))
    act = ef > ef.max() * 10 ** (-40 / 20)
    if act.sum() < 4:
        return 99.0
    d = float(np.mean(db(eb[act]) - db(ef[act])))
    return round(d - max(0.0, excess), 1)


# ----------------------------------------------------------------------------
# Sugerencias automáticas (se convierten en los valores de los controles)
# ----------------------------------------------------------------------------
def suggest(an, profile, art_db):
    pr = PROFILES.get(profile, PROFILES["otro"])
    guitar = profile.startswith("guitarra")
    dist = profile == "guitarra_distorsion"
    o = {int(k): v for k, v in an["octaves"].items()}
    tg = pr["target"]

    nr = _interp(an["noise_db"], [-95, -80, -70, -60, -50], [15, 30, 50, 70, 85])
    nr = max(nr, _interp(an.get("hash_db", 0), [0.5, 2, 4, 8], [20, 50, 70, 85]))
    art = _interp(art_db, [0.3, 1, 2, 4], [30, 50, 70, 85])

    # EQ correctiva
    def dev(fc):
        return o.get(fc, 0) - tg.get(fc, 0)

    low_g = float(np.clip(-dev(125) * 0.4, -4, 3))
    mud = dev(250)
    mud_g = float(np.clip(-(mud - 2) * 0.6, -5, 0)) if mud > 2 else 0.0
    mid_g = float(np.clip(-dev(500) * 0.3, -3, 2))
    pres = dev(2000)
    pres_g = float(np.clip(-pres * 0.4, -3, 3))
    harsh = dev(4000)
    harsh_g = float(np.clip(-(harsh - 2) * 0.6, -5, 0)) if harsh > 2 else 0.0
    air = dev(8000)
    air_g = float(np.clip(-air * 0.35, -2, 3))

    hpf = pr["hpf"]
    if o.get(63, 0) < -25 and hpf < 50:
        hpf = 45
    # restauración espectral
    holes, inst = an.get("holes_pct", 0), an.get("instab_pct", 0)
    damage = max(holes, inst / 4)
    fill = _interp(damage, [0.2, 1.5, 5], [30, 60, 90])
    cut = an.get("hf_cut", an["bandwidth"])
    hf_base = {"guitarra_limpia": 65, "guitarra_acustica": 65, "guitarra_distorsion": 30, "bajo": 0, "voz": 55,
               "bateria": 60, "teclados": 55, "otro": 50}.get(profile, 50)
    hf = 0 if cut >= 19000 else hf_base * _interp(cut, [11000, 16000, 18500], [1.0, 0.85, 0.5])
    # si a menudo hay agudos reales por encima del corte, no es un corte de códec: no sintetizar
    hf *= _interp(an.get("hf_beyond_pct", 0), [10, 30], [1.0, 0.0])
    severe = damage > 1.5 or art_db > 1.5 or cut < 16500

    lpf = 20000
    if hf == 0 and an["bandwidth"] < 17000:
        lpf = int(min(20000, an["bandwidth"] * 1.1))
    if dist:
        lpf = min(lpf, 16000)

    plr = an["plr"]
    squashed = max(_interp(plr, [7, 10, 14], [1.0, 0.5, 0.1]),
                   _interp(an["dyn_range"], [4, 8, 14], [1.0, 0.5, 0.1]))
    att_amt = (45 if not dist else 20) * squashed
    exp_amt = (40 if not dist else 20) * squashed

    target_slope = SLOPE_TARGET.get(profile, -4.5)
    tilt = float(np.clip(-(an["slope"] - target_slope) * 0.7, -2.5, 2.5))
    stereo = an.get("channels", 2) == 2
    flip = stereo and an["polarity"] < 0
    mono_bass = stereo and (an["corr_low"] < 0.6 or profile in ("bajo", "bateria"))

    # limpieza de cuantización: cuanto más daño, más cantidad de "Artefactos"
    art = max(art, _interp(an.get("q_lines", 0), [0.5, 2, 5], [40, 65, 85]),
              _interp(an.get("q_grain", 0), [5, 20, 40], [35, 60, 80]))
    # acumulación en medios-agudos: corte estático + EQ dinámica vigilando esa zona
    bu = an.get("buildup", [])
    buildup_on, deh_f1, e1, deh_f2, e2 = pick_buildup(bu)
    if buildup_on:
        harsh_g = min(harsh_g, -min(6.0, (e1 - 3.0) * 0.6))  # la EQ dinámica hace el resto
        harsh_f = deh_f1
    else:
        harsh_f = 4500
    crisp = {"guitarra_distorsion": 15, "bateria": 30, "bajo": 10}.get(profile, 25)
    if severe:
        crisp += 15

    p = {
        "rep.fill": round(fill), "rep.hf": round(hf),
        "rep.bal": 100 if an.get("balance_max_db", 0) < -1 else 0,
        "dyn.crisp": crisp,
        "deh.on": True, "deh.f1": round(deh_f1), "deh.f2": round(deh_f2), "deh.depth": 9 if (severe or buildup_on) else 5, "deh.sens": 3 if buildup_on else 4,
        "deh.r1": an.get("deh_r", {}).get(str(round(deh_f1)), 99), "deh.r2": an.get("deh_r", {}).get(str(round(deh_f2)), 99),
        "eq.tilt": round(tilt, 1), "eq.slope_target": target_slope,
        "stereo.flip": bool(flip), "stereo.align": int(an["lag"]) if stereo else 0,
        "stereo.monobass": bool(mono_bass), "stereo.monofreq": 150 if an["corr_low"] < 0.3 else 120,
        "clean.nr": round(nr), "clean.art": round(art),
        "hpf.freq": hpf, "hpf.slope": 24 if profile in ("voz", "guitarra_distorsion") else 12,
        "gate.on": True,
        "gate.thr": round(float(np.clip(min(an["noise_db"] + 8, an["loud95"] - 30), -85, -35))),
        "gate.ratio": 3 if dist else 2, "gate.range": 25 if dist else 15,
        "gate.att": 0.5 if profile == "bateria" else 2, "gate.rel": 200 if profile == "guitarra_acustica" else 120,
        "gate.hold": 20,
        "eq.on": True,
        "eq.low_g": round(low_g, 1), "eq.low_f": 100,
        "eq.mud_g": round(mud_g, 1), "eq.mud_f": 300,
        "eq.mid_g": round(mid_g, 1), "eq.mid_f": 800,
        "eq.pres_g": round(pres_g, 1), "eq.pres_f": 2500,
        "eq.harsh_g": round(harsh_g, 1), "eq.harsh_f": round(harsh_f),
        "eq.air_g": round(air_g, 1), "eq.air_f": 10000,
        "eq.lpf": lpf,
        "notches": [dict(r, on=True) for r in an["resonances"]],
        "dyn.on": True, "dyn.attack": round(att_amt), "dyn.sustain": 0, "dyn.expand": round(exp_amt),
        "comp.on": True, "comp.thr": round(float(np.clip(an["loud95"] - 1, -40, -3))),
        "comp.ratio": 2.0, "comp.att": 25, "comp.rel": 200, "comp.knee": 6, "comp.makeup": 0,
        "mic.on": True, "mic.amount": (30 if buildup_on else 60) if guitar else (25 if buildup_on else 40),
        "mic.excite": (8 if hf > 0 else round(25 if air < -4 else 10)) if not dist else 5,
        "mic.sat": 15 if guitar else 8,
        "stereo.width": 100,
        "rev.on": False, "rev.type": "estudio", "rev.mix": 15, "rev.size": 0.8, "rev.predelay": 8, "rev.tone": 55,
        "out.gain": 0, "out.ceil": -1.0, "out.level": "tp",
    }
    return p


def make_spectrograms(workdir, before, after, sr):
    pb = pa = None
    for c in range(before.shape[0]):
        _, _, zb = signal.stft(before[c], sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)
        _, _, za = signal.stft(after[c], sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)
        b_ = restore.pool_columns(np.abs(zb)); a_ = restore.pool_columns(np.abs(za))
        pb = b_ if pb is None else pb + b_
        pa = a_ if pa is None else pa + a_
    restore.write_spectrograms(pb / before.shape[0], pa / before.shape[0], sr, workdir)


BUFFERS = ("x", "n", "a", "na", "r", "h", "e")


def apply_edits(workdir, edits):
    """Rehace las versiones actuales a partir de las originales + lista de ediciones manuales."""
    base = os.path.join(workdir, "base")
    if not os.path.isdir(base):
        os.makedirs(base)
        for k in BUFFERS:
            os.replace(os.path.join(workdir, f"{k}.wav"), os.path.join(base, f"{k}.wav"))
    cur = {}
    for k in BUFFERS:
        y, sr = sf.read(os.path.join(base, f"{k}.wav"), dtype="float32", always_2d=True)
        y = y.T.copy()
        for e in edits:
            y = restore.apply_edit(y, sr, e)
        sf.write(os.path.join(workdir, f"{k}.wav"), y.T, sr, subtype="FLOAT")
        cur[k] = y
    xb, sr = sf.read(os.path.join(base, "x.wav"), dtype="float32", always_2d=True)
    make_spectrograms(workdir, xb.T, cur["na"] + cur["r"] + cur["h"] + cur["e"], sr)


# ----------------------------------------------------------------------------
# Proceso completo
# ----------------------------------------------------------------------------
def process(in_path, workdir, profile, progress):
    progress(2, "Leyendo el archivo de audio…")
    x, sr = load_audio(in_path, workdir)
    n = x.shape[1]
    progress(6, "Quitando DC y reparando picos recortados…")
    x = remove_dc(x, sr)
    x, clips = declip(x)

    progress(10, "Analizando ruido, dinámica y espectro…")
    an = analyze(x, sr)
    an["clips_fixed"] = clips
    an["sr"] = sr
    an["channels"] = int(x.shape[0])
    an["duration"] = round(n / sr, 2)

    outs = {k: np.zeros_like(x) for k in ("n", "a", "na", "r", "h", "e")}
    bal = []
    hashes = []
    art_dbs, holes, inst, song, clone, metal, qstats = [], [], [], [], [], set(), []
    nch = x.shape[0]
    cutoff = float(an["bandwidth"])

    def to_time(Z):
        _, y = signal.istft(Z, sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)
        y = y[:n]
        return np.pad(y, (0, n - len(y))).astype(np.float32) if len(y) < n else y.astype(np.float32)

    # --- análisis y decisiones en ESTÉREO ENLAZADO: las mismas máscaras y reparaciones para
    #     ambos canales (cada uno conserva su fase) -> no se descorrelaciona la imagen estéreo
    progress(14, "Análisis espectral (estéreo enlazado)…")
    Zs = []
    for c in range(nch):
        _, _, z = signal.stft(x[c], sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)
        Zs.append(z.astype(np.complex64))
    mag_l = np.sqrt(np.mean([np.abs(z) ** 2 for z in Zs], axis=0)).astype(np.float32)
    prof = noise_profile(mag_l)
    progress(20, "Detectando ruido, hash digital, artefactos y zumbidos metálicos…")
    mn, ma, art_db = spectral_masks(mag_l, sr, prof)
    fr_ = np.linspace(0, sr / 2, mag_l.shape[0]) > 2000
    e0 = mag_l[fr_] ** 2; e1 = (mag_l[fr_] * mn[fr_]) ** 2
    kk = max(1, int(10 * sr / HOP)); nseg = max(1, mag_l.shape[1] // kk)
    hashes.append(float(max(10 * np.log10((e0[:, i * kk:(i + 1) * kk].sum() + 1e-12) / (e1[:, i * kk:(i + 1) * kk].sum() + 1e-12)) for i in range(nseg))))
    del e0, e1
    mm, mf = restore.metallic_mask(mag_l, sr)
    ma *= mm[:, None]
    metal.update(mf)
    on = restore.onsets(mag_l, sr)
    progress(26, "Limpiando líneas, manchas y bloques de cuantización…")
    mq, dmg, qs = restore.dequantize_mask(mag_l * mn * ma, sr, cutoff, on)
    ma *= mq
    qstats.append(qs)
    art_dbs.append(art_db)
    M = mn * ma
    zc_l = mag_l * M
    progress(32, "Buscando en la canción las partes que suenan bien…")
    R_l, hp, ip, sp, cp = restore.repair_fill(
        zc_l, sr, cutoff, on, dmg,
        progress=lambda f: progress(32 + 30 * f, f"Clonando partes buenas sobre las dañadas… {int(f * 100)}%"))
    holes.append(hp); inst.append(ip); song.append(sp); clone.append(cp)
    Gm1, gcurve = adaptive_balance(zc_l + R_l, sr)
    bal.append(float(gcurve.min()) if len(gcurve) else 0.0)
    sm_l = uniform_filter(zc_l, size=(9, 9)) + 1e-9
    for c in range(nch):
        tag = f"Canal {c + 1}/{nch}" if nch > 1 else "Audio"
        z = Zs[c]
        zc = z * M
        # reparto de la reconstrucción enlazada según el peso de este canal, con su fase
        ratio = np.clip(uniform_filter(np.abs(zc), size=(9, 9)) / sm_l, 0, np.sqrt(nch) * 1.5)
        R = (R_l * ratio * np.exp(1j * np.angle(zc))).astype(np.complex64)
        progress(64 + 22 * c / nch, f"{tag}: sintetizando agudos perdidos…")
        H = restore.hf_synth(to_time(zc + R), zc_l + R_l, sr, cutoff, on)
        E = ((zc + R) * Gm1).astype(np.complex64)
        for i, (k, Z) in enumerate((("n", z * mn), ("a", z * ma), ("na", zc), ("r", R), ("h", H), ("e", E))):
            progress(64 + 22 * c / nch + 22 / nch * (0.2 + 0.13 * i), f"{tag}: generando audio restaurado ({i + 1}/6)…")
            outs[k][c] = to_time(Z)
        del zc, R, H, E, ratio
    del Zs, mag_l, mn, ma, M, zc_l, R_l, Gm1, sm_l

    art_db = float(np.mean(art_dbs)) if art_dbs else 0.0
    an["art_db"] = round(art_db, 2)
    an["noise_removed_db"] = round(float(db(np.sqrt(np.mean(x ** 2))) - db(np.sqrt(np.mean(outs["n"] ** 2)))), 2)
    an["holes_pct"] = round(float(np.mean(holes)), 2)
    an["instab_pct"] = round(float(np.mean(inst)), 1)
    an["song_pct"] = round(float(np.mean(song)), 1)
    an["clone_pct"] = round(float(np.mean(clone)), 1)
    an["balance_max_db"] = round(float(min(bal)) if bal else 0.0, 1)
    an["hash_db"] = round(float(np.mean(hashes)) if hashes else 0.0, 2)
    if qstats and qstats[0]:
        an["q_lines"] = round(float(np.mean([q.get("lines", 0) for q in qstats])), 2)
        an["q_spots"] = round(float(np.mean([q.get("spots", 0) for q in qstats])), 2)
        an["q_bursts"] = int(np.sum([q.get("bursts", 0) for q in qstats]))
        an["q_grain"] = round(float(np.mean([q.get("grain", 0) for q in qstats])), 1)
    an["metal_tones"] = sorted(metal)[:8]
    an["hf_cut"] = round(cutoff)

    progress(88, "Guardando versiones restauradas…")
    sf.write(os.path.join(workdir, "x.wav"), x.T, sr, subtype="FLOAT")
    for k, v in outs.items():
        sf.write(os.path.join(workdir, f"{k}.wav"), v.T, sr, subtype="FLOAT")
    progress(92, "Dibujando espectrogramas…")
    make_spectrograms(workdir, x, outs["na"] + outs["r"] + outs["h"] + outs["e"], sr)

    progress(95, "Comprobando la zona de estridencia (2-8 kHz)…")
    est = (outs["na"] + 0.8 * outs["r"] + 0.6 * outs["h"] + outs["e"]).mean(axis=0)
    st = SLOPE_TARGET.get(profile, -4.5)
    bu = buildup(est, sr, st, cutoff)
    an["buildup"] = [(round(f), round(e, 1)) for f, e in bu[:6]]
    an["buildup_orig"] = round(max([e for _, e in buildup(x.mean(axis=0), sr, st, cutoff)] or [0]), 1)
    on_, f1, e1, f2, e2 = pick_buildup(bu)
    an["deh_r"] = {str(round(f1)): deharsh_ref(est, sr, f1, max(0.0, e1 - 3.0) if on_ else 0.0),
                   str(round(f2)): deharsh_ref(est, sr, f2, max(0.0, e2 - 3.0) if on_ else 0.0)}
    progress(96, "Calculando ajustes automáticos…")
    params = suggest(an, profile, art_db)
    result = dict(analysis=an, params=params, profile=profile)
    with open(os.path.join(workdir, "result.json"), "w") as fh:
        json.dump(result, fh)
    progress(100, "¡Listo!")
    return result
