#!/bin/bash
# Instalador del Mejorador de Stems (solo hace falta una vez).
cd "$(dirname "$0")" || exit 1
echo "==> Creando entorno de Python…"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
echo "==> Instalando dependencias (puede tardar unos minutos)…"
.venv/bin/pip install -q -r requirements.txt
if ! command -v ffmpeg >/dev/null && [ ! -x /usr/local/bin/ffmpeg ] && [ ! -x /opt/homebrew/bin/ffmpeg ]; then
  echo "⚠️  No se encuentra ffmpeg (necesario para MP3). Instálalo con:  brew install ffmpeg"
fi
echo ""
echo "✅ Instalación terminada."
"$(pwd)/Iniciar Mejorador Stems.command"
