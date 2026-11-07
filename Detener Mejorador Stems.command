#!/bin/bash
# Detiene el servidor del Mejorador de Stems.
cd "$(dirname "$0")" || exit 1
pkill -f "MEJORADOR STEMS/server.py" 2>/dev/null || pkill -f "python -W ignore server.py --no-browser" 2>/dev/null
for P in $(seq 8915 8919); do
  PID=$(lsof -ti tcp:$P -sTCP:LISTEN 2>/dev/null); [ -n "$PID" ] && ps -p $PID -o command= | grep -q server.py && kill $PID
done
echo "Mejorador de Stems detenido."
