#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serveur fm_drow GPU — à lancer SUR un pod cloud (RunPod ou équivalent), pas en local.

Contexte : fm_drow (Kokoro-82M fine-tuné FR) produit une friture systématique en
inférence CPU sur ce PC (plancher de bruit ~2500x plus élevé qu'en GPU, mesuré et
confirmé). Ce serveur reprend le même loader (fm_drow.py) mais sur device="cuda",
qui a été validé propre pendant l'entraînement sur RunPod.

────────────────────────────────────────────────────────────────────────
INSTALLATION SUR LE POD (une fois par pod)
────────────────────────────────────────────────────────────────────────

  1. Uploader sur le pod, côte à côte :
       fm_drow_pod_server.py   (ce fichier)
       fm_drow_kokoro/         (le package voix : fm_drow.pth, config.json,
                                 voices/fm_drow.pt, fm_drow.py)

  2. pip install kokoro misaki espeakng-loader num2words soundfile flask flask-cors "misaki[en]"
     python -m spacy download en_core_web_sm

  3. Dans l'UI RunPod, exposer le port HTTP choisi ci-dessous (PORT) pour ce pod
     -> RunPod fournit une URL proxy stable du type :
        https://<pod-id>-<port>.proxy.runpod.net
     Cette URL est à coller dans EPUBSON (section "🚀 Pod GPU").

────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────

  python fm_drow_pod_server.py

  Ctrl+C pour arrêter (ou éteins le pod depuis l'UI RunPod pour cesser d'être facturé).

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

# Dossier du package voix, à côté de ce script par défaut (override possible via env var)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DROW_DIR = os.environ.get('FM_DROW_DIR', os.path.join(SCRIPT_DIR, 'fm_drow_kokoro'))
sys.path.insert(0, DROW_DIR)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PYTHONUTF8', '1')

HOST = '0.0.0.0'  # écoute sur toutes les interfaces (le pod route via son proxy)
PORT = int(os.environ.get('PORT', 5008))
SAMPLE_RATE = 24000
DEVICE = os.environ.get('FM_DROW_DEVICE', 'cuda')  # 'cuda' sur le pod, 'cpu' en secours

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def add_pna_header(response):
    response.headers['Access-Control-Allow-Private-Network'] = 'true'
    return response


_pipeline = None
_voice = None
_lock = threading.Lock()
_device_used = None


def load_model():
    global _pipeline, _voice, _device_used
    print(f"  Chargement fm_drow sur device='{DEVICE}'…", flush=True)
    try:
        import torch
        device = DEVICE if (DEVICE != 'cuda' or torch.cuda.is_available()) else 'cpu'
        if device != DEVICE:
            print(f"  ! CUDA indisponible, repli sur '{device}' (friture probable)")
        from fm_drow import load
        _pipeline, _voice = load(device=device)
        # Bug kokoro : isinstance(voice, torch.FloatTensor) ne reconnaît que les
        # tenseurs CPU. Un voicepack chargé directement sur 'cuda' échappe à ce
        # test et fait planter load_voice(). On le garde en CPU ; le pipeline le
        # déplace lui-même sur le bon device via .to(model.device) à chaque appel.
        if device == 'cuda':
            _voice = _voice.cpu()
        _device_used = device
        print(f"  OK fm_drow charge sur '{device}'")
        if device == 'cuda':
            print(f"  GPU : {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"  X Erreur chargement : {e}")
        import traceback
        traceback.print_exc()


def float32_to_wav(samples) -> bytes:
    import numpy as np
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


VOICES = [{'name': 'fm_drow', 'gender': 'M', 'language': 'fr', 'custom': True}]


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok' if _pipeline is not None else 'loading',
        'voices': VOICES if _pipeline is not None else [],
        'device': _device_used,
        'model': 'Kokoro-82M fine-tune (fm_drow) — GPU',
    })


@app.route('/voices', methods=['GET'])
def voices_endpoint():
    return jsonify({'voices': VOICES if _pipeline is not None else []})


@app.route('/tts', methods=['POST', 'OPTIONS'])
def tts():
    if request.method == 'OPTIONS':
        return ('', 204)

    if _pipeline is None:
        return jsonify({'error': 'Modele pas encore charge, reessaie dans quelques secondes.'}), 503

    data = request.get_json(force=True, silent=True) or {}
    text = (data.get('text') or '').strip()
    speed = float(data.get('speed', 1.0))

    if not text:
        return jsonify({'error': 'text vide'}), 400

    try:
        import numpy as np
        with _lock:
            chunks = [a for _gs, _ps, a in _pipeline(text, voice=_voice, speed=speed)]

        if not chunks:
            return jsonify({'error': 'Synthese vide'}), 500

        audio = np.concatenate(chunks).astype('float32')
        wav_bytes = float32_to_wav(audio)

        return send_file(
            io.BytesIO(wav_bytes),
            mimetype='audio/wav',
            as_attachment=False,
            download_name='fm_drow.wav',
        )
    except Exception as e:
        print(f"X Erreur synthese : {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/shutdown', methods=['POST', 'OPTIONS'])
def shutdown():
    if request.method == 'OPTIONS':
        return ('', 204)

    def delayed_exit():
        time.sleep(0.3)
        os._exit(0)

    threading.Thread(target=delayed_exit, daemon=True).start()
    return jsonify({'status': 'shutting down'})


@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'fm_drow GPU pod server',
        'port': PORT,
        'device': _device_used,
        'endpoints': ['/health (GET)', '/voices (GET)', '/tts (POST)', '/shutdown (POST)'],
        'ready': _pipeline is not None,
    })


if __name__ == '__main__':
    print('-' * 60)
    print('  fm_drow GPU pod server')
    print('-' * 60)
    load_model()
    print(f"\nOK Serveur en ecoute sur http://{HOST}:{PORT}")
    print(f"   (expose ce port dans l'UI RunPod pour obtenir l'URL proxy)")
    print(f"   Ctrl+C pour arreter -- eteins le pod pour cesser d'etre facture.\n")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
