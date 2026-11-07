#!/usr/bin/env python3
"""Restaurador automático de audio (sin abrir la app).

Uso:
  .venv/bin/python restaurar.py ARCHIVO.wav [MÁS ARCHIVOS…] [opciones]

Opciones:
  -o, --salida CARPETA     carpeta de destino (por defecto: «Resultados» junto a este script)
  -p, --perfil PERFIL      guitarra_limpia (defecto), guitarra_distorsion, guitarra_acustica,
                           bajo, voz, bateria, teclados, otro
  --nivel tp|match|none    tp = picos a −1 dBTP (defecto) · match = mismo volumen que el
                           original (stems de una misma canción) · none = sin normalizar
  --bits 24|32             WAV de 24 bits (defecto) o 32 bits coma flotante
  --reverb                 añade la reverb de sala de estudio (por defecto, stem seco)

Pipeline (el mismo que la app con los ajustes automáticos):
  1. Análisis: ruido, compresión, cuantización, corte de agudos, estridencia por tramos, fase.
  2. Limpieza espectral adaptativa: siseo constante + hash digital que aparece solo en algunos
     tramos, chisporroteo de IA, zumbidos metálicos, líneas finas, manchas y bloques.
  3. Reparación: clona partes buenas de la canción sobre las dañadas, rellena huecos de códec
     sin adelantar energía a los ataques y, solo si hay corte real de agudos, sintetiza el aire.
  4. Equilibrio adaptativo de los tramos estridentes y protección de la zona 2-8 kHz.
  5. Paso alto ≥ 80 Hz, EQ correctiva (barro 300-500 Hz, resonancias), EQ dinámica.
  6. Dinámica: ataques y nitidez de púa, compresor suave 2:1 con ataque medio y release natural.
  7. Carácter de micro de estudio, fase/estéreo (graves en mono si se cancelan), limitador.
  8. Normalización a −1 dBTP (pico verdadero) y WAV 24 bits a la frecuencia original (≥ 44,1 kHz).
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy import signal

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import engine  # noqa: E402
import pipeline  # noqa: E402


def bar(f, txt, width=28):
    n = int(f * width)
    sys.stdout.write(f"\r  [{'█' * n}{'·' * (width - n)}] {int(f * 100):3d}%  {txt[:60]:60s}")
    sys.stdout.flush()


def report(an, P):
    lines = []
    add = lines.append
    add(f"  Formato: {an['sr']} Hz · {'estéreo' if an['channels'] == 2 else 'mono'} · {an['duration']:.1f} s")
    add(f"  Ruido de fondo {an['noise_db']} dBFS · hash digital en agudos {an.get('hash_db', 0)} dB → reducción {P['clean.nr']}%")
    add(f"  Cuantización: líneas {an.get('q_lines', 0)}% · bloques {an.get('q_grain', 0)}% → limpieza de artefactos {P['clean.art']}%")
    cut = an.get("hf_cut", an["bandwidth"]) / 1000
    add(f"  Corte de agudos {cut:.1f} kHz (agudos reales por encima el {an.get('hf_beyond_pct', 0)}% del tiempo) → síntesis {P['rep.hf']}%")
    add(f"  Huecos de códec {an.get('holes_pct', 0)}% · cortes en ladrillo {an.get('instab_pct', 0)}% · clonado de partes buenas {an.get('clone_pct', 0)}%")
    add(f"  Tramos estridentes: hasta {an.get('balance_max_db', 0)} dB de corrección (equilibrio adaptativo {P.get('rep.bal', 0)}%)")
    bu = [b for b in an.get("buildup", []) if b[0] >= 2000]
    if bu:
        add(f"  Zona 2-8 kHz: mayor exceso +{bu[0][1]} dB en {bu[0][0]} Hz → EQ {P['eq.harsh_g']} dB en {P['eq.harsh_f']} Hz + EQ dinámica {P['deh.f1']}/{P['deh.f2']} Hz")
    add(f"  Dinámica: rango {an['dyn_range']} dB → ataque {P['dyn.attack']}%, nitidez {P.get('dyn.crisp', 0)}%, descompresión {P['dyn.expand']}%, compresor {P['comp.ratio']}:1")
    add(f"  EQ: paso alto {P['hpf.freq']} Hz · barro {P['eq.mud_g']} dB en {P['eq.mud_f']} Hz · inclinación {P['eq.tilt']} dB · resonancias {len(P['notches'])}")
    if an["channels"] == 2:
        add(f"  Fase: correlación {an['corr']} (graves {an['corr_low']}) → graves en mono {'sí' if P['stereo.monobass'] else 'no'}"
            f"{', polaridad invertida' if P['stereo.flip'] else ''}{', alineado ' + str(an['lag_ms']) + ' ms' if P['stereo.align'] else ''}")
    return "\n".join(lines)


def process_file(path, out_dir, profile, level, bits, with_reverb):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n▶ {os.path.basename(path)}")
    work = tempfile.mkdtemp(prefix="restaurar_")
    t0 = time.time()
    try:
        res = engine.process(path, work, profile, lambda p, s: bar(p / 100 * 0.75, s))
        P = res["params"]
        an = res["analysis"]
        P["out.level"] = level
        P["hpf.freq"] = max(80, P["hpf.freq"]) if profile.startswith("guitarra") else P["hpf.freq"]
        P["eq.mud_f"] = min(max(P["eq.mud_f"], 300), 500)
        P["eq.mud_g"] = min(P["eq.mud_g"], -1.5) if profile.startswith("guitarra") else P["eq.mud_g"]
        P["comp.ratio"] = min(max(P["comp.ratio"], 2.0), 3.0)
        if with_reverb:
            P["rev.on"] = True
        B = {k: sf.read(os.path.join(work, f"{k}.wav"), dtype="float32", always_2d=True)[0].T for k in engine.BUFFERS}
        sr = an["sr"]
        y = pipeline.render(B, P, sr, lambda f, s: bar(0.75 + f * 0.25, s))
        n = B["x"].shape[1]
        y = y[:, :n]
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, f"{name}_restaurado.wav")
        if bits == 24:
            yd = y + (np.random.random(y.shape) - np.random.random(y.shape)) / 2 ** 23  # dither TPDF
            sf.write(dest, np.clip(yd, -1, 1).T, sr, subtype="PCM_24")
        else:
            sf.write(dest, y.T, sr, subtype="FLOAT")
        # espectrogramas antes / después para comparar
        spec_dir = os.path.join(out_dir, f"{name}_espectrogramas")
        os.makedirs(spec_dir, exist_ok=True)
        engine.make_spectrograms(spec_dir, B["x"], y, sr)
        for v in ("antes", "despues", "mapa"):
            os.replace(os.path.join(spec_dir, f"spec_{v}.png"), os.path.join(spec_dir, f"{v}.png"))
        bar(1, "Listo")
        tp = 20 * np.log10(pipeline.true_peak(y, sr) + 1e-12)
        print(f"\n{report(an, P)}")
        print(f"  Salida: {dest}\n  Pico verdadero {tp:.2f} dBTP · {bits} bits · {sr} Hz · {n / sr:.1f} s (misma duración que el original)"
              f" · {time.time() - t0:.0f} s de proceso")
        return dest
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Restaura audio muy comprimido o con artefactos de IA.")
    ap.add_argument("archivos", nargs="+")
    ap.add_argument("-o", "--salida", default=os.path.join(BASE, "Resultados"))
    ap.add_argument("-p", "--perfil", default="guitarra_limpia", choices=list(engine.PROFILES))
    ap.add_argument("--nivel", default="tp", choices=["tp", "match", "none"])
    ap.add_argument("--bits", type=int, default=24, choices=[24, 32])
    ap.add_argument("--reverb", action="store_true")
    a = ap.parse_args()
    for f in a.archivos:
        if not os.path.isfile(f):
            print(f"✗ No existe: {f}")
            continue
        process_file(f, a.salida, a.perfil, a.nivel, a.bits, a.reverb)


if __name__ == "__main__":
    main()
