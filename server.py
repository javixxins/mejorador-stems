"""Servidor local del Mejorador de Stems.

Puerto 8915 (si estuviera ocupado prueba 8916-8919). No usa los puertos de las
otras apps (5000, 5050, 7000, 8008, 8731, 8790, 8850, 8862).
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser

from flask import Flask, jsonify, request, send_file, send_from_directory

import glob

import numpy as np
import soundfile as sf

import engine

BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, "trabajo")
CONFIG = os.path.join(BASE, "config.json")
PORTS = range(8915, 8920)
os.makedirs(WORK, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = None
JOBS = {}


def load_cfg():
    try:
        with open(CONFIG) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_cfg(cfg):
    with open(CONFIG, "w") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)


def cleanup_old():
    now = time.time()
    for d in os.listdir(WORK):
        p = os.path.join(WORK, d)
        if os.path.isdir(p) and now - os.path.getmtime(p) > 2 * 86400:
            shutil.rmtree(p, ignore_errors=True)


@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/api/config")
def config():
    return jsonify(app="mejorador-stems", version="1.0")


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No se ha recibido ningún archivo"), 400
    name = f.filename or "audio.wav"
    ext = os.path.splitext(name)[1].lower() or ".wav"
    jid = uuid.uuid4().hex[:12]
    d = os.path.join(WORK, jid)
    os.makedirs(d)
    path = os.path.join(d, "input" + ext)
    f.save(path)
    JOBS[jid] = dict(dir=d, input=path, name=name, progress=0, stage="Archivo subido", done=False, error=None)
    return jsonify(id=jid, name=name)


@app.route("/api/analyze/<jid>", methods=["POST"])
def analyze(jid):
    job = JOBS.get(jid)
    if not job:
        return jsonify(error="Archivo no encontrado, súbelo otra vez"), 404
    profile = (request.get_json(silent=True) or {}).get("profile", "guitarra_limpia")
    job.update(progress=0, stage="Empezando…", done=False, error=None, result=None)

    def prog(p, s):
        job["progress"] = round(float(p), 1)
        job["stage"] = s

    def run():
        try:
            job["result"] = engine.process(job["input"], job["dir"], profile, prog)
            job["done"] = True
        except Exception as e:  # noqa
            traceback.print_exc()
            job["error"] = f"No se pudo procesar el audio: {e}"

    threading.Thread(target=run, daemon=True).start()
    return jsonify(ok=True)


@app.route("/api/job/<jid>")
def job_status(jid):
    job = JOBS.get(jid)
    if not job:
        return jsonify(error="Trabajo no encontrado"), 404
    return jsonify({k: job.get(k) for k in ("progress", "stage", "done", "error", "result", "name")})


@app.route("/api/audio/<jid>/<which>")
def audio(jid, which):
    if which not in engine.BUFFERS:
        return "no", 404
    job = JOBS.get(jid)
    if not job:
        return "no", 404
    return send_file(os.path.join(job["dir"], which + ".wav"), mimetype="audio/wav", conditional=True)


@app.route("/api/spec/<jid>/<which>")
def spec(jid, which):
    job = JOBS.get(jid)
    if not job or which not in ("antes", "despues", "mapa"):
        return "no", 404
    return send_file(os.path.join(job["dir"], f"spec_{which}.png"), mimetype="image/png", max_age=0)


# --- edición manual por zonas (espectrograma) ---------------------------------
@app.route("/api/edit/<jid>", methods=["POST"])
def edit(jid):
    job = JOBS.get(jid)
    if not job or not job.get("done"):
        return jsonify(error="Analiza el archivo primero"), 400
    e = request.get_json(silent=True) or {}
    try:
        e = dict(t0=float(e["t0"]), t1=float(e["t1"]), f0=float(e["f0"]), f1=float(e["f1"]),
                 mode=e.get("mode", "atenuar"), gain=float(e.get("gain", -12)))
    except Exception:
        return jsonify(error="Selección no válida"), 400
    job.setdefault("edits", []).append(e)
    engine.apply_edits(job["dir"], job["edits"])
    return jsonify(ok=True, edits=len(job["edits"]))


@app.route("/api/undo/<jid>", methods=["POST"])
def undo(jid):
    job = JOBS.get(jid)
    if not job or not job.get("edits"):
        return jsonify(ok=True, edits=0)
    job["edits"].pop()
    engine.apply_edits(job["dir"], job["edits"])
    return jsonify(ok=True, edits=len(job["edits"]))


# --- puente con iZotope RX ------------------------------------------------------
RX_DIR = os.path.expanduser("~/Music/Mejorador Stems - RX")


def rx_app():
    apps = sorted(glob.glob("/Applications/iZotope RX*Audio Editor.app") + glob.glob("/Applications/iZotope RX*/iZotope RX*Audio Editor.app"))
    return apps[-1] if apps else None


@app.route("/api/rx/info")
def rx_info():
    a = rx_app()
    return jsonify(app=os.path.basename(a)[:-4] if a else None, folder=RX_DIR)


def _rx_open(path):
    a = rx_app()
    if not a:
        return False
    subprocess.Popen(["open", "-a", a, path])
    return True


@app.route("/api/rx/send/<jid>", methods=["POST"])
def rx_send(jid):
    """kind=original: manda el audio original (WAV 24 bits) a RX y vigila cuándo se guarda.
    kind=final: recibe el WAV ya procesado (cuerpo de la petición) y lo abre en RX."""
    job = JOBS.get(jid)
    if not job:
        return jsonify(error="Archivo no encontrado"), 404
    kind = request.args.get("kind", "original")
    os.makedirs(RX_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(job["name"]))[0]
    if kind == "original":
        dest = os.path.join(RX_DIR, f"{stem} - para RX.wav")
        src = os.path.join(job["dir"], "x.wav")  # incluye las reparaciones manuales hechas aquí
        y, sr = sf.read(src, dtype="float32")
        sf.write(dest, np.clip(y, -1, 1), sr, subtype="PCM_24")
        job["rx"] = dict(path=dest, mtime=os.path.getmtime(dest), size=-1)
    else:
        dest = os.path.join(RX_DIR, f"{stem} - resultado final.wav")
        with open(dest, "wb") as fh:
            fh.write(request.get_data())
    ok = _rx_open(dest)
    return jsonify(path=dest, opened=ok, app=os.path.basename(rx_app() or "")[:-4] or None)


@app.route("/api/rx/status/<jid>")
def rx_status(jid):
    job = JOBS.get(jid)
    rx = job and job.get("rx")
    if not rx or not os.path.exists(rx["path"]):
        return jsonify(watching=False)
    mt, sz = os.path.getmtime(rx["path"]), os.path.getsize(rx["path"])
    changed = mt > rx["mtime"] + 0.5
    stable = changed and sz == rx["size"]  # mismo tamaño en dos consultas seguidas = terminó de guardar
    rx["size"] = sz
    return jsonify(watching=True, changed=changed, ready=stable, path=rx["path"])


@app.route("/api/rx/import/<jid>", methods=["POST"])
def rx_import(jid):
    job = JOBS.get(jid)
    rx = job and job.get("rx")
    if not rx:
        return jsonify(error="No hay ningún archivo enviado a RX"), 400
    nid = uuid.uuid4().hex[:12]
    d = os.path.join(WORK, nid)
    os.makedirs(d)
    path = os.path.join(d, "input.wav")
    shutil.copy(rx["path"], path)
    name = job["name"]
    JOBS[nid] = dict(dir=d, input=path, name=name, progress=0, stage="Importado desde RX", done=False, error=None,
                     rx=dict(path=rx["path"], mtime=os.path.getmtime(rx["path"]), size=-1))
    rx["mtime"] = os.path.getmtime(rx["path"])
    return jsonify(id=nid, name=name)


def _osa(script_lines):
    args = ["osascript"]
    for line in script_lines:
        args += ["-e", line]
    r = subprocess.run(args, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


@app.route("/api/choose-folder", methods=["POST"])
def choose_folder():
    cfg = load_cfg()
    last = cfg.get("last_folder") or os.path.expanduser("~/Desktop")
    if not os.path.isdir(last):
        last = os.path.expanduser("~")
    last = last.replace('"', '\\"')
    code, out, err = _osa([
        "activate",
        f'set f to choose folder with prompt "Elige la carpeta donde guardar el stem mejorado" default location (POSIX file "{last}")',
        "POSIX path of f",
    ])
    if code != 0:
        if "-128" in err:  # cancelado por el usuario
            return jsonify(path=None, cancelled=True)
        return jsonify(path=None, error=err or "No se pudo abrir el selector de carpetas")
    cfg["last_folder"] = out
    save_cfg(cfg)
    return jsonify(path=out)


@app.route("/api/save", methods=["POST"])
def save():
    """Recibe el WAV renderizado. format=wav lo guarda tal cual; format=mp3 lo
    convierte a MP3 320 kbps con ffmpeg."""
    folder = request.args.get("folder", "")
    fmt = request.args.get("format", "wav").lower()
    name = os.path.basename(request.args.get("name", "stem_mejorado")) or "stem_mejorado"
    if not folder or not os.path.isdir(folder):
        return jsonify(error="La carpeta no existe"), 400
    name = os.path.splitext(name)[0] + (".mp3" if fmt == "mp3" else ".wav")
    dest = os.path.join(folder, name)
    base, ext = os.path.splitext(dest)
    i = 2
    while os.path.exists(dest):
        dest = f"{base} ({i}){ext}"
        i += 1
    tmp = os.path.join(WORK, f"export_{uuid.uuid4().hex[:8]}.wav")
    with open(tmp if fmt == "mp3" else dest, "wb") as fh:
        while True:
            chunk = request.stream.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    if fmt == "mp3":
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-codec:a", "libmp3lame",
                            "-b:a", "320k", dest], capture_output=True, text=True)
        os.remove(tmp)
        if r.returncode != 0:
            return jsonify(error="ffmpeg no pudo crear el MP3: " + r.stderr[-300:]), 500
    return jsonify(path=dest)


@app.route("/api/reveal", methods=["POST"])
def reveal():
    path = (request.get_json(silent=True) or {}).get("path", "")
    if os.path.exists(path):
        subprocess.Popen(["open", "-R", path])
    return jsonify(ok=True)


def free_port():
    for p in PORTS:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise SystemExit("Los puertos 8915-8919 están ocupados")


if __name__ == "__main__":
    cleanup_old()
    port = free_port()
    with open(os.path.join(BASE, ".port"), "w") as fh:
        fh.write(str(port))
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"Mejorador de Stems en http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
