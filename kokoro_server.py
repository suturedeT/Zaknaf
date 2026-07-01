#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serveur Kokoro TTS local — voix fm_drow (française, homme, fine-tuné).

────────────────────────────────────────────────────────────────────────
INSTALLATION (une seule fois)
────────────────────────────────────────────────────────────────────────

  pip install kokoro misaki espeakng-loader num2words soundfile flask flask-cors

  (espeak-ng est inclus via espeakng-loader sur Windows)

────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────

  python kokoro_server.py

  Serveur sur http://127.0.0.1:5007
  Ouvre epub-to-audiobook.html (ou epub-reader.html) → voix "fm_drow" disponible.
  Ctrl+C pour arrêter.

────────────────────────────────────────────────────────────────────────
"""
import io
import os
import sys
import wave
import threading
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

DROW_DIR = r"C:\Users\EA_ADM\Documents\claude_ai\drizzt_out\fm_drow_kokoro"
sys.path.insert(0, DROW_DIR)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTHONUTF8", "1")

HOST = "127.0.0.1"
PORT = 5007
SAMPLE_RATE = 24000

VOICES = [
    {"name": "fm_drow", "gender": "M", "label": "Drow (fine-tuné maison)"},
]

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_pna_header(response):
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

_pipeline = None
_voice    = None
_lock     = threading.Lock()


def load_model():
    global _pipeline, _voice
    print("  Chargement fm_drow (peut prendre 10-20 s)…", flush=True)
    try:
        from fm_drow import load
        _pipeline, _voice = load(device="cpu")
        print("  ✓ fm_drow chargé — prêt !")
    except Exception as e:
        print(f"  ✗ Erreur chargement : {e}")
        import traceback; traceback.print_exc()


def float32_to_wav(samples) -> bytes:
    """Convertit un array float32 numpy en bytes WAV PCM 16-bit."""
    import numpy as np
    pcm   = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if _pipeline is not None else "loading",
        "voices": VOICES if _pipeline is not None else [],
        "model": "Kokoro-82M fine-tuné (fm_drow)",
    })


@app.route("/voices", methods=["GET"])
def voices():
    return jsonify({"voices": VOICES if _pipeline is not None else []})


@app.route("/tts", methods=["POST", "OPTIONS"])
def tts():
    if request.method == "OPTIONS":
        return ("", 204)

    if _pipeline is None:
        return jsonify({"error": "Modèle pas encore chargé, réessaie dans quelques secondes."}), 503

    data  = request.get_json(force=True, silent=True) or {}
    text  = (data.get("text") or "").strip()
    speed = float(data.get("speed", 1.0))

    if not text:
        return jsonify({"error": "text vide"}), 400

    try:
        import numpy as np
        with _lock:
            chunks = [a for _gs, _ps, a in _pipeline(text, voice=_voice, speed=speed)]

        if not chunks:
            return jsonify({"error": "Synthèse vide"}), 500

        audio     = np.concatenate(chunks).astype("float32")
        wav_bytes = float32_to_wav(audio)

        return send_file(
            io.BytesIO(wav_bytes),
            mimetype="audio/wav",
            as_attachment=False,
            download_name="kokoro.wav",
        )
    except Exception as e:
        print(f"✗ Erreur synthèse : {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/shutdown", methods=["POST", "OPTIONS"])
def shutdown():
    if request.method == "OPTIONS":
        return ("", 204)
    def delayed_exit():
        time.sleep(0.3)
        os._exit(0)
    threading.Thread(target=delayed_exit, daemon=True).start()
    return jsonify({"status": "shutting down"})


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "Kokoro TTS local (fm_drow)",
        "port": PORT,
        "endpoints": ["/health (GET)", "/voices (GET)", "/tts (POST)", "/shutdown (POST)"],
        "ready": _pipeline is not None,
    })


if __name__ == "__main__":
    print("─" * 60)
    print("  Kokoro TTS local — voix fm_drow")
    print("─" * 60)
    load_model()
    print(f"\n✓ Serveur en écoute sur http://{HOST}:{PORT}\n")
    print("  Ouvre epub-to-audiobook.html (ou epub-reader.html) → voix 'fm_drow' disponible.")
    print("  Ctrl+C pour arrêter.\n")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
