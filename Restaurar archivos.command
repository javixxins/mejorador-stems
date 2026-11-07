#!/bin/bash
# Restaura uno o varios archivos de audio sin abrir la app (pipeline automático completo).
cd "$(dirname "$0")" || exit 1
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
[ -x .venv/bin/python ] || ./instalar.command
FILES=$(osascript -e 'set fs to choose file with prompt "Elige los audios a restaurar (WAV o MP3)" of type {"public.audio"} with multiple selections allowed' -e 'set out to ""' -e 'repeat with f in fs' -e 'set out to out & POSIX path of f & linefeed' -e 'end repeat' -e 'return out' 2>/dev/null)
[ -z "$FILES" ] && { echo "Cancelado."; exit 0; }
PERFIL=$(osascript -e 'choose from list {"guitarra_limpia","guitarra_distorsion","guitarra_acustica","bajo","voz","bateria","teclados","otro"} with prompt "Tipo de instrumento:" default items {"guitarra_limpia"}' 2>/dev/null)
[ "$PERFIL" = "false" ] && exit 0
ARGS=()
while IFS= read -r f; do [ -n "$f" ] && ARGS+=("$f"); done <<< "$FILES"
.venv/bin/python -W ignore restaurar.py "${ARGS[@]}" -p "${PERFIL:-guitarra_limpia}"
open Resultados
echo ""
echo "✅ Terminado. Los archivos restaurados están en la carpeta Resultados."
