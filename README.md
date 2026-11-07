# Mejorador de Stems

App local (macOS) para limpiar y restaurar stems generados por IA (Suno y similares) o audio muy comprimido, y dejarlos listos para uso profesional.

Analiza el archivo, propone todos los ajustes automáticamente y permite retocarlos con **preescucha en tiempo real** y comparación **Antes / Después** con volumen igualado. Incluye un script para procesar archivos por lotes sin abrir la app.

## Qué hace

- **Limpieza espectral adaptativa**: siseo constante y "hash" digital que aparece solo en algunos tramos, chisporroteo de IA, zumbidos metálicos, líneas finas, manchas y bloques de cuantización. Respeta los armónicos y los ataques.
- **Reparación**: clona sobre las partes dañadas el sonido de partes sanas de la misma canción donde se toca lo mismo, rellena huecos de códec sin adelantar energía a los ataques (sin pre-eco) y, solo si hay corte real de agudos, sintetiza el aire perdido.
- **Equilibrio adaptativo**: suaviza los tramos mucho más estridentes que las partes sanas y protege la zona de 2-8 kHz (EQ fija suave + EQ dinámica).
- **EQ correctiva**: paso alto, barro, resonancias persistentes, inclinación tonal.
- **Dinámica**: nitidez del ataque de púa, descompresión, compresor de ataque medio y release natural.
- **Carácter de micro de estudio**, excitador y saturación suaves.
- **Fase y estéreo**: polaridad, alineación de canales, graves en mono (Linkwitz-Riley), anchura, medidor de correlación.
- **Reverb** opcional (estudio, cámara, sala, placa).
- **Espectrograma** antes / después / mapa de reparaciones, con reparación manual de zonas (tipo *Spectral Repair*).
- Puente con **iZotope RX**: abre el audio en RX y lo reimporta automáticamente al guardar.
- Exportación en **WAV 24 / 32 bits o MP3 320 kbps**, normalizado a **−1 dBTP** o al mismo nivel que el original, con la misma duración y frecuencia de muestreo (los stems siguen sincronizados).

## Instalación

Requisitos: macOS, Python 3.9+ y [ffmpeg](https://ffmpeg.org) (`brew install ffmpeg`).

```bash
./instalar.command
```

Crea el entorno `.venv` e instala las dependencias de `requirements.txt`.

## Uso

**App**: doble clic en `Mejorador Stems.app` (o en `Iniciar Mejorador Stems.command`). Arranca el servidor en `http://127.0.0.1:8915` y abre el navegador. Para pararlo: `Detener Mejorador Stems.command`.

> El icono `.app` lleva escrita la ruta de la carpeta. Si mueves el proyecto, edita `APP_DIR` en `Mejorador Stems.app/Contents/MacOS/launcher`.

**Por lotes, sin abrir la app**: doble clic en `Restaurar archivos.command`, o desde Terminal:

```bash
.venv/bin/python restaurar.py "mi_stem.wav" [-p guitarra_limpia] [--nivel tp|match|none] [--bits 24|32] [--reverb]
```

Los resultados se guardan en `Resultados/`, con espectrogramas de antes y después.

Perfiles: `guitarra_limpia`, `guitarra_distorsion`, `guitarra_acustica`, `bajo`, `voz`, `bateria`, `teclados`, `otro`.

## Cómo funciona

| Archivo | Función |
|---|---|
| `engine.py` | Carga, análisis y máscaras espectrales en estéreo enlazado. Genera las versiones `x, n, a, na, r, h, e`. |
| `restore.py` | Reparación: ataques, cuantización, clonado por similitud musical, síntesis de agudos, espectrogramas, edición manual. |
| `server.py` | Servidor Flask local (puerto 8915): subida, progreso, audio, espectrogramas, guardado, puente con RX. |
| `index.html`, `static/worklets.js` | Interfaz y cadena de audio en tiempo real (Web Audio + AudioWorklet, latencia cero). |
| `pipeline.py` | La misma cadena de mezcla en Python. |
| `restaurar.py` | Script de línea de comandos que une análisis, restauración y mezcla. |

Como la cantidad de cada limpieza o reparación es lineal, el navegador mezcla las versiones que calcula el servidor con pesos simples: mover un control equivale exactamente a recalcular, pero el cambio se oye al instante.
