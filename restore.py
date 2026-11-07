"""Restauración espectral de audio muy comprimido (MP3 agresivo / IA).

  * repair_fill   – rellena huecos espectrales y caídas repentinas del ancho de
                    banda ("inestabilidad espectral", sonido sumergido/metálico)
  * hf_synth      – síntesis de agudos por encima del corte del códec (aire/brillo)
  * metallic_mask – atenúa tonos agudos fijos que no pertenecen al instrumento
  * spectrogram   – imágenes antes / después / mapa de reparaciones
  * apply_edit    – reparación manual de una zona (tiempo x frecuencia)

Todo devuelve componentes ADITIVOS (R, H) o máscaras, para que el navegador
pueda dosificarlos en tiempo real con un simple control de ganancia.
"""
import struct
import zlib

import numpy as np
from scipy import signal
from scipy.ndimage import binary_dilation, binary_opening, median_filter, uniform_filter, uniform_filter1d

EPS = 1e-10
NFFT, HOP = 2048, 512


def db(x):
    return 20 * np.log10(np.maximum(x, EPS))


# ----------------------------------------------------------------------------
# Tonos metálicos fijos en agudos
# ----------------------------------------------------------------------------
def metallic_mask(mag, sr):
    nb, nf = mag.shape
    freqs = np.linspace(0, sr / 2, nb)
    mask = np.ones(nb, np.float32)
    b0 = int(np.searchsorted(freqs, 3500))
    if b0 >= nb - 20 or nf < 20:
        return mask, []
    Ld = db(mag[b0:, ::4])
    loud = Ld.max(axis=0) > Ld.max() - 50
    if loud.sum() < 10:
        return mask, []
    Ld = Ld[:, loud]
    env = median_filter(Ld, size=(15, 1), mode="nearest")
    ex = Ld - env
    peaky = ex > 8
    frac = peaky.mean(axis=1)
    found = []
    for k in np.flatnonzero(frac > 0.55):
        if k > 0 and frac[k - 1] > frac[k] or k < len(frac) - 1 and frac[k + 1] > frac[k]:
            continue  # solo el máximo local
        exc = float(np.mean(ex[k][peaky[k]]))
        att = 10 ** (-min(12.0, exc * 0.8) / 20)
        for j in (k - 1, k, k + 1):
            if 0 <= j < len(frac):
                mask[b0 + j] = min(mask[b0 + j], att if j == k else np.sqrt(att))
        found.append(round(float(freqs[b0 + k])))
    return mask, found[:12]


# ----------------------------------------------------------------------------
# Ataques (onsets): se protegen en todas las etapas para no emborronarlos
# ----------------------------------------------------------------------------
def onsets(mag, sr):
    nb = mag.shape[0]
    freqs = np.linspace(0, sr / 2, nb)
    b = (freqs > 150) & (freqs < 4000)
    e = np.log(np.sum(mag[b] ** 2, axis=0) + 1e-10)
    flux = np.maximum(0, np.diff(e, prepend=e[0]))
    med = median_filter(flux, size=31, mode="nearest")
    pk, _ = signal.find_peaks(flux, height=med + 0.6, distance=4)
    on = np.zeros(mag.shape[1], bool)
    on[pk] = True
    return on


def near(on, before, after):
    """frames a menos de `before` frames antes o `after` después de un ataque."""
    out = on.copy()
    for s in range(1, before + 1):
        out |= np.roll(on, -s)
    for s in range(1, after + 1):
        out |= np.roll(on, s)
    return out


# ----------------------------------------------------------------------------
# Artefactos de cuantización del códec
# ----------------------------------------------------------------------------
def dequantize_mask(mag, sr, cutoff, on):
    """Máscara que elimina: líneas finas cortas (pitidos), manchas aisladas,
    estallidos verticales sin respaldo musical y textura en bloques/granulada.
    Devuelve (mascara, daño_por_frame[0..1], stats)."""
    nb, nf = mag.shape
    freqs = np.linspace(0, sr / 2, nb)
    M = np.ones((nb, nf), np.float32)
    b0 = int(np.searchsorted(freqs, 1500))
    b1 = int(np.searchsorted(freqs, min(cutoff * 1.02, sr / 2 * 0.99)))
    if b1 - b0 < 16 or nf < 16:
        return M, np.zeros(nf, np.float32), {}
    L = db(mag[b0:b1]).astype(np.float32)
    live = L.max(axis=0) > L.max() - 70
    prot = near(on, 1, 2)

    # a) líneas finas cortas: picos estrechos que duran < 8 frames (~90 ms)
    medf = median_filter(L, size=(9, 1), mode="nearest")
    P = (L - medf) > 8
    longt = binary_dilation(binary_opening(P, structure=np.ones((1, 8), bool)), structure=np.ones((3, 1), bool))
    after_on = near(on, 1, 8)
    short = P & ~longt & live[None, :] & ~after_on[None, :]
    # b) manchas aisladas (píxeles sueltos muy por encima de su entorno)
    med33 = median_filter(L, size=(3, 3), mode="nearest")
    spot = ((L - med33) > 10) & ~longt & live[None, :]
    tgt = np.where(short, medf + 2, L)
    tgt = np.where(spot, np.minimum(tgt, med33 + 2), tgt)

    # c) estallidos verticales en agudos sin subida de la zona media (no son ataques)
    hb = (freqs[b0:b1] > 3000)
    if hb.any():
        e_hf = 10 * np.log10(np.mean(10 ** (L[hb] / 10), axis=0) + 1e-20)
        midb = (freqs > 400) & (freqs < 2000)
        e_mid = 10 * np.log10(np.mean(mag[midb] ** 2, axis=0) + 1e-20)
        burst = e_hf - median_filter(e_hf, size=9, mode="nearest")
        mid_up = e_mid - median_filter(e_mid, size=9, mode="nearest")
        vb = (burst > 6) & (mid_up < 3) & ~prot & live
        cut = np.where(vb, burst - 3, 0)
        tgt[hb] = np.where(vb[None, :], np.minimum(tgt[hb], L[hb] - cut[None, :]), tgt[hb])
    else:
        vb = np.zeros(nf, bool)

    # d) textura en bloques / granulada: acerca cada píxel a su entorno suave
    #    (en frecuencia siempre; en tiempo solo lejos de los ataques)
    hb2 = freqs[b0:b1] > 3000
    sm_f = uniform_filter1d(tgt, 3, axis=0)
    sm = uniform_filter1d(sm_f, 3, axis=1)
    sm = np.where(prot[None, :], sm_f, sm)
    rough = np.abs(tgt - sm) > 4
    blend = np.where(rough & hb2[:, None] & live[None, :] & ~longt, 0.5, 0.0)
    tgt = tgt + (sm - tgt) * blend

    m = np.clip(10 ** ((tgt - L) / 20), 0.05, 2.0)
    M[b0:b1] = uniform_filter(m, size=(3, 1))
    dmg = np.clip((short.mean(axis=0) * 8 + spot.mean(axis=0) * 8 + vb * 0.5), 0, 1)
    stats = dict(lines=float(100 * short[:, live].mean()) if live.any() else 0.0,
                 spots=float(100 * spot[:, live].mean()) if live.any() else 0.0,
                 bursts=int(vb.sum()), grain=float(100 * rough[:, live].mean()) if live.any() else 0.0)
    return M, dmg.astype(np.float32), stats


# ----------------------------------------------------------------------------
# Referencia "según la canción": momentos donde se toca lo mismo
# ----------------------------------------------------------------------------
def song_reference(mag, sr, b0, b1, k=8, min_sim=0.88, progress=None, good=None):
    """Para cada frame busca los k frames más parecidos de TODA la canción
    (comparando 150-3000 Hz, la zona menos dañada por el códec) y devuelve la
    mediana de su espectro en los bins b0:b1, ajustada al nivel del frame.
    Devuelve (ref_db  [bins x frames] o None donde no hay parecido, % frames con pareja)."""
    nb, nf = mag.shape
    freqs = np.linspace(0, sr / 2, nb)
    edges = 150 * (3000 / 150) ** (np.arange(49) / 48)
    idx = np.searchsorted(freqs, edges)
    feat = np.stack([np.sqrt(np.mean(mag[idx[i]:max(idx[i + 1], idx[i] + 1)] ** 2, axis=0)) for i in range(48)])
    F = db(feat).T.astype(np.float32)  # frames x 48
    lvl = F.mean(axis=1)
    Fc = F - lvl[:, None]
    Fc /= np.linalg.norm(Fc, axis=1, keepdims=True) + 1e-6
    live = lvl > lvl.max() - 50
    L = db(mag[b0:b1]).astype(np.float32)
    ref = np.full(L.shape, np.nan, np.float32)
    cand = np.flatnonzero(live & (good if good is not None else True))  # solo partes que suenan bien
    if len(cand) < 4 * k:
        return None, 0.0
    Fcand = Fc[cand]
    matched = 0
    chunk = 512
    for c0 in range(0, nf, chunk):
        rows = np.arange(c0, min(nf, c0 + chunk))
        sim = Fc[rows] @ Fcand.T  # chunk x cand
        # no usar el propio entorno (±8 frames): buscamos OTRA aparición
        near = np.abs(rows[:, None] - cand[None, :]) <= 8
        sim[near] = -1
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        tsim = np.take_along_axis(sim, top, axis=1)
        for r, t in enumerate(rows):
            if not live[t]:
                continue
            good = top[r][tsim[r] >= min_sim]
            if len(good) < 3:
                continue
            nn = cand[good]
            ref[:, t] = np.median(L[:, nn] + (lvl[t] - lvl[nn])[None, :], axis=1)
            matched += 1
        if progress:
            progress(min(1.0, (c0 + chunk) / nf))
    return ref, 100.0 * matched / max(1, live.sum())


# ----------------------------------------------------------------------------
# Relleno de huecos y caídas de ancho de banda
# ----------------------------------------------------------------------------
def repair_fill(zc, sr, cutoff, on, dmg, progress=None):
    """Reconstruye lo dañado. zc: STFT ya limpia; on: ataques; dmg: daño por frame.
    - Rellena huecos y caídas del ancho de banda (sin tocar los instantes previos a un ataque).
    - Clona: los frames muy dañados se rehacen con el espectro de otras partes BUENAS de la
      canción donde suena lo mismo.
    Devuelve (R aditivo, % huecos, % inestabilidad, % guiado por la canción, % clonado)."""
    mag = np.abs(zc)
    nb, nf = mag.shape
    freqs = np.linspace(0, sr / 2, nb)
    R = np.zeros(zc.shape, np.float32 if np.isrealobj(zc) else np.complex64)
    b0 = int(np.searchsorted(freqs, 1500))
    b1 = int(np.searchsorted(freqs, min(cutoff, sr / 2 * 0.98)))
    if b1 - b0 < 8 or nf < 24:
        return R, 0.0, 0.0, 0.0, 0.0
    band = mag[b0:b1]
    L = db(band)

    # referencia CAUSAL: solo lo que sonó antes (así nunca se adelanta energía a un ataque)
    ref_s = median_filter(L, size=(1, 9), mode="nearest", origin=(0, 4))
    dec = 4
    nfd = nf // dec
    Ld = L[:, : nfd * dec].reshape(b1 - b0, nfd, dec).max(axis=2)
    ref_l = np.repeat(median_filter(Ld, size=(1, 7), mode="nearest", origin=(0, 3)), dec, axis=1)
    if ref_l.shape[1] < nf:
        ref_l = np.pad(ref_l, ((0, 0), (0, nf - ref_l.shape[1])), mode="edge")
    ref = np.maximum(ref_s, ref_l - 3)
    del ref_s, ref_l, Ld

    good = dmg <= 0.2  # solo partes claramente sanas sirven de modelo
    sref, song_pct = song_reference(mag, sr, b0, b1, progress=progress, good=good)

    body_b = (freqs >= 400) & (freqs < 1500)
    body = db(np.sqrt(np.mean(mag[body_b] ** 2, axis=0)))
    body_ref = median_filter(body, size=41, mode="nearest")
    body_past = median_filter(body, size=9, mode="nearest", origin=4)  # lo que sonaba justo antes
    drop = np.maximum(0, np.maximum(body_ref, body_past) - body)
    alive = (drop < 9) & (body > body.max() - 50)  # nunca rellenar cuando el sonido cae o hay silencio
    pre = near(on, 4, 0) & ~on  # instantes previos a un ataque: prohibido añadir energía

    has = ~np.isnan(sref) if sref is not None else np.zeros_like(L, bool)
    if sref is not None:
        filled_ref = np.where(has, sref, L)
        sm = uniform_filter(filled_ref, size=(3, 3))
        sref = np.where(has, np.where(near(on, 0, 1)[None, :], uniform_filter1d(filled_ref, 3, axis=0), sm), np.nan)
        ref = np.where(has, np.maximum(sref, ref - 6), ref)
    deficit = (ref - L) - drop[None, :]
    D = (deficit > 12) & alive[None, :] & ~pre[None, :]
    # Un hueco de códec es ESTRECHO (franja que falta con contenido arriba y abajo) o
    # está sobre un corte en "ladrillo". Una caída suave de agudos entre rasgueos es
    # natural y no se rellena.
    nband = max(8, int(560 / (sr / NFFT)))
    longrun = binary_opening(D, structure=np.ones((nband, 1), bool))
    narrow = D & ~longrun
    Ls = uniform_filter1d(L, 5, axis=0)
    w12 = max(3, int(140 / (sr / NFFT)))
    steep = np.zeros_like(D)
    steep[w12:-w12] = (Ls[:-2 * w12] - Ls[2 * w12:]) > 24  # caída >24 dB en ~280 Hz
    fb = freqs[b0:b1]
    steep &= (fb > 3000)[:, None]
    has_wall = steep.any(axis=0)
    wall_k = np.where(has_wall, np.argmax(steep, axis=0), L.shape[0])
    above_wall = np.arange(L.shape[0])[:, None] >= wall_k[None, :]
    hole = narrow | (D & above_wall)
    hole &= np.roll(hole, 1, 0) | np.roll(hole, -1, 0)
    w = np.clip((deficit - 12) / 6, 0, 1) * hole
    w = uniform_filter(w.astype(np.float32), size=(3, 1))
    w = np.minimum(w, uniform_filter1d(w, 3, axis=1, origin=1))  # sin extender hacia atrás
    target = 10 ** ((ref - 2 - drop[None, :]) / 20)
    add = (np.maximum(0, target - band) * w).astype(np.float32)

    # clonación de partes buenas sobre frames dañados (sube Y baja: también quita basura)
    clone_pct = 0.0
    if sref is not None:
        cw = np.clip((dmg - 0.45) / 0.3, 0, 1)  # solo instantes claramente dañados
        cw = uniform_filter1d(cw, 5)
        cw = np.where(pre, 0, cw) * alive
        cmask = has & (cw[None, :] > 0)
        clone_mag = 10 ** ((np.where(has, sref, L) - drop[None, :] * 0) / 20)
        clone_mag = np.minimum(clone_mag, 10 ** (ref / 20) * 1.12)  # nunca por encima de la referencia
        delta = np.where(cmask, (clone_mag - band) * cw[None, :], 0).astype(np.float32)
        clone_pct = float(100 * (cw > 0.05).sum() / max(1, alive.sum()))
    else:
        delta = 0
    del target, deficit, ref

    # fase: siempre la del propio audio en ese punto (una fase inventada genera
    # batidos entre bins vecinos = textura granulada)
    ph = np.angle(zc[b0:b1])
    comp = add + delta
    comp = uniform_filter1d(comp, 3, axis=0)
    comp_t = uniform_filter1d(comp, 3, axis=1, origin=1)  # promedia con el pasado, nunca con el futuro
    comp = np.where(near(on, 0, 1)[None, :], comp, comp_t)
    R[b0:b1] = comp if np.isrealobj(R) else comp * np.exp(1j * ph).astype(np.complex64)

    act = alive & (body > body.max() - 45)
    holes_pct = float(100 * hole[:, act].mean()) if act.any() else 0.0
    Lb = db(band)
    thr_db = Lb.max() - 60
    above = Lb > thr_db
    top = np.where(above.any(axis=0), (b1 - b0) - np.argmax(above[::-1], axis=0), 0)
    inst_pct = float(100 * (has_wall & act).sum() / max(1, act.sum()))  # frames con corte en ladrillo
    return R, holes_pct, inst_pct, song_pct, clone_pct


# ----------------------------------------------------------------------------
# Síntesis de agudos (extensión de ancho de banda)
# ----------------------------------------------------------------------------
def hf_synth(y_clean, zc, sr, cutoff, on):
    """Extensión de ancho de banda tipo excitador: los armónicos nuevos se generan
    rectificando en el TIEMPO la última octava real (conserva los ataques exactos y
    la textura natural, sin líneas inarmónicas ni grano), y luego se moldean con una
    envolvente suave que continúa la pendiente del instrumento."""
    nb, nf = zc.shape
    H = np.zeros(zc.shape, np.complex64)
    top = min(sr / 2 * 0.97, 20000)
    if cutoff > top - 1200 or cutoff < 4000:
        return H
    nyq = sr / 2
    b, a = signal.butter(4, [0.45 * cutoff / nyq, 0.95 * cutoff / nyq], btype="bandpass")
    band = signal.filtfilt(b, a, y_clean)
    gen = np.abs(band)  # armónicos pares: la 2ª cae justo en la octava perdida
    b, a = signal.butter(6, 0.9 * cutoff / nyq, btype="highpass")
    gen = signal.filtfilt(b, a, gen).astype(np.float32)
    _, _, G = signal.stft(gen, sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)
    G = G[:, :nf].astype(np.complex64)
    if G.shape[1] < nf:
        G = np.pad(G, ((0, 0), (0, nf - G.shape[1])))

    df = sr / NFFT
    kC = int(cutoff / df)
    k0 = int(0.9 * cutoff / df)
    k_top = int(top / df)
    mag = np.abs(zc)
    ltas = db(np.sqrt(np.mean(mag ** 2, axis=1)))
    ks = np.arange(int(kC * 0.5), int(kC * 0.95))
    slope = float(np.polyfit(np.log2(ks * df), ltas[ks], 1)[0]) if len(ks) > 4 else -9.0
    slope = float(np.clip(slope, -24, -4)) - 3.0
    e_top = np.sqrt(np.mean(mag[int(0.75 * kC):int(0.95 * kC)] ** 2, axis=0))
    f_ref = 0.85 * cutoff
    fk = np.arange(k0, k_top) * df
    target = e_top[None, :] * ((fk / f_ref) ** (slope / 6.02))[:, None]
    gm = np.abs(G[k0:k_top])
    env = uniform_filter(gm, size=(15, 3)) + 1e-9  # envolvente suave del material generado
    gain = target / env
    need = np.clip(1 - mag[k0:k_top] / (target + 1e-9), 0, 1)
    Hs = G[k0:k_top] * (gain * need).astype(np.float32)
    # nivel estable en el tiempo (sin grano): promedio con el pasado, salvo en los ataques
    hm = np.abs(Hs)
    sm = np.sqrt(np.maximum(0, uniform_filter1d(uniform_filter1d(hm ** 2, 3, axis=0), 7, axis=1, origin=3)))
    att = near(on, 0, 1)[None, :]
    new_m = np.where(att, np.maximum(hm, sm), sm)
    new_m = np.minimum(new_m, target * need)  # nunca más de lo que toca en ese instante (sin colas)
    H[k0:k_top] = Hs / (hm + 1e-12) * new_m
    return H


# ----------------------------------------------------------------------------
# Espectrogramas (PNG sin dependencias)
# ----------------------------------------------------------------------------
_CMAP = np.array([[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
                  [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]], np.float32)


def _cmap(v):
    v = np.clip(np.nan_to_num(v, nan=0.0), 0, 1) * (len(_CMAP) - 1)
    i = np.minimum(v.astype(int), len(_CMAP) - 2)
    t = (v - i)[..., None]
    return _CMAP[i] * (1 - t) + _CMAP[i + 1] * t


def _png(path, rgb):
    h, w, _ = rgb.shape
    a = np.clip(rgb, 0, 255).astype(np.uint8)
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


SPEC_W, SPEC_H, SPEC_FMIN = 1600, 300, 40.0


def pool_columns(mag, width=SPEC_W):
    """Reduce (bins x frames) a (bins x width) cogiendo el máximo por columna."""
    nb, nf = mag.shape
    edges = np.linspace(0, nf, min(width, nf) + 1).astype(int)
    return np.maximum.reduceat(mag, edges[:-1], axis=1).astype(np.float32)


def _rows(sr, nb):
    nyq = sr / 2
    f = SPEC_FMIN * (nyq / SPEC_FMIN) ** (np.arange(SPEC_H)[::-1] / (SPEC_H - 1))
    return np.clip(np.round(f / nyq * (nb - 1)).astype(int), 0, nb - 1)


def write_spectrograms(before, after, sr, workdir):
    rows = _rows(sr, before.shape[0])
    B = db(before[rows])
    A = db(after[rows])
    top = np.percentile(B, 99.7)
    rng = 90.0
    _png(f"{workdir}/spec_antes.png", _cmap((B - top + rng) / rng))
    _png(f"{workdir}/spec_despues.png", _cmap((A - top + rng) / rng))
    # mapa: gris = audio, rojo = eliminado, verde = añadido/reconstruido
    g = np.clip((A - top + rng) / rng, 0, 1)[..., None] * 150
    img = np.repeat(g, 3, axis=2)
    d = A - B
    vis = (np.maximum(A, B) > top - 70)
    rem = np.clip(-d / 12, 0, 1) * vis
    addm = np.clip(d / 9, 0, 1) * vis
    img[..., 0] += rem * 230
    img[..., 1] += addm * 210 - rem * 60
    img[..., 2] += addm * 60 - rem * 60
    _png(f"{workdir}/spec_mapa.png", img)


# ----------------------------------------------------------------------------
# Edición manual de una zona (como el "Spectral Repair" de RX)
# ----------------------------------------------------------------------------
def apply_edit(y, sr, e):
    """y: (canales, muestras). e: dict(t0, t1, f0, f1, mode='atenuar'|'reparar', gain)."""
    n = y.shape[1]
    pad = 3 * NFFT
    s0 = max(0, int(e["t0"] * sr) - pad)
    s1 = min(n, int(e["t1"] * sr) + pad)
    if s1 - s0 < 2 * NFFT:
        return y
    seg = y[:, s0:s1]
    f, t, Z = signal.stft(seg, sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP, axis=-1)
    tt = t + s0 / sr
    fi = (f >= e["f0"]) & (f <= e["f1"])
    ti = (tt >= e["t0"]) & (tt <= e["t1"])
    if not fi.any() or not ti.any():
        return y
    # bordes suaves (3 bins / 3 frames)
    wf = uniform_filter1d(fi.astype(np.float32), 5)
    wt = uniform_filter1d(ti.astype(np.float32), 5)
    W = np.outer(wf, wt)[None]
    if e.get("mode") == "reparar":
        idx = np.flatnonzero(ti)
        a0, a1 = max(0, idx[0] - 4), min(len(tt), idx[-1] + 5)
        before = np.abs(Z[..., max(0, a0 - 4):a0]).mean(axis=-1) if a0 > 0 else None
        after = np.abs(Z[..., a1:a1 + 4]).mean(axis=-1) if a1 < len(tt) else None
        if before is None:
            before = after
        if after is None:
            after = before
        if before is None:
            return y
        span = np.clip((np.arange(len(tt)) - a0) / max(1, a1 - a0), 0, 1)
        interp = before[..., None] * (1 - span) + after[..., None] * span
        mag = np.abs(Z)
        new_mag = np.minimum(mag, interp)  # nunca añade energía: solo quita lo que sobra
        Z = Z * (1 - W) + (Z / (mag + EPS) * new_mag) * W
    else:
        g = 10 ** (float(e.get("gain", -12)) / 20)
        Z = Z * (1 - W * (1 - g))
    _, out = signal.istft(Z, sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP, time_axis=-1, freq_axis=-2)
    out = out[:, : s1 - s0]
    keep0 = NFFT if s0 > 0 else 0
    keep1 = (s1 - s0) - (NFFT if s1 < n else 0)
    y = y.copy()
    y[:, s0 + keep0:s0 + keep1] = out[:, keep0:keep1]
    return y
