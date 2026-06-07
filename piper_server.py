#!/usr/bin/env python3
"""
Serveur Piper TTS local pour ÉpubSon.

────────────────────────────────────────────────────────────────────────
INSTALLATION (une seule fois)
────────────────────────────────────────────────────────────────────────

  1. Installer Python 3.9+ (déjà fait si tu as utilisé Whisper)

  2. Installer les dépendances :
       pip install piper-tts flask flask-cors

  3. Télécharger les 2 modèles depuis :
       https://huggingface.co/rhasspy/piper-voices/tree/main/fr/fr_FR

     a) fr_FR-upmc-medium :
        https://huggingface.co/rhasspy/piper-voices/tree/main/fr/fr_FR/upmc/medium
        → télécharger fr_FR-upmc-medium.onnx + fr_FR-upmc-medium.onnx.json

     b) fr_FR-mls-medium :
        https://huggingface.co/rhasspy/piper-voices/tree/main/fr/fr_FR/mls/medium
        → télécharger fr_FR-mls-medium.onnx + fr_FR-mls-medium.onnx.json

  4. Placer les 4 fichiers dans le dossier `models/` à côté de ce script :
       Zaknaf/
         ├── epub-to-audiobook.html
         ├── piper_server.py
         └── models/
             ├── fr_FR-upmc-medium.onnx
             ├── fr_FR-upmc-medium.onnx.json
             ├── fr_FR-mls-medium.onnx
             └── fr_FR-mls-medium.onnx.json

────────────────────────────────────────────────────────────────────────
USAGE (à chaque session)
────────────────────────────────────────────────────────────────────────

  Lance simplement :
       python piper_server.py

  Le serveur démarre sur http://127.0.0.1:5005.
  Laisse-le tourner dans un terminal. Ouvre epub-to-audiobook.html
  dans ton navigateur → le backend "Piper local" sera disponible.

  Pour arrêter : Ctrl+C dans le terminal.

────────────────────────────────────────────────────────────────────────
"""
import io
import os
import sys
import wave

# Force UTF-8 sur la console Windows (cp1252 par défaut ne gère pas ─, ✓, ⚠, etc.)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

try:
    from piper import PiperVoice
    from piper.config import SynthesisConfig
except ImportError:
    print("✗ piper-tts non installé. Lance : pip install piper-tts flask flask-cors")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
HOST = '127.0.0.1'
PORT = 5005

# Voix à charger au démarrage. Ajoute ici si tu télécharges d'autres modèles.
# Note : fr_FR-upmc-medium est multi-locuteur (jessica=0, pierre=1).
VOICE_NAMES = [
    'fr_FR-upmc-medium',
]

# ── Init Flask + CORS ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

voices = {}


def load_voices():
    """Charge tous les modèles déclarés dans VOICE_NAMES."""
    if not os.path.exists(MODELS_DIR):
        print(f"✗ Dossier models/ introuvable : {MODELS_DIR}")
        print(f"  Crée-le et place les fichiers .onnx + .onnx.json dedans.")
        return
    for name in VOICE_NAMES:
        onnx = os.path.join(MODELS_DIR, f'{name}.onnx')
        cfg = os.path.join(MODELS_DIR, f'{name}.onnx.json')
        if not os.path.exists(onnx) or not os.path.exists(cfg):
            print(f"⚠ {name} : fichiers manquants ({onnx})")
            continue
        try:
            print(f"  Chargement {name}…", flush=True)
            voices[name] = PiperVoice.load(onnx, config_path=cfg)
            print(f"  ✓ {name} chargé")
        except Exception as e:
            print(f"  ✗ {name} : erreur de chargement → {e}")


@app.route('/health', methods=['GET'])
def health():
    """Ping pour que l'app web sache si le serveur est dispo."""
    return jsonify({
        'status': 'ok',
        'voices': list(voices.keys()),
        'count': len(voices),
    })


@app.route('/tts', methods=['POST', 'OPTIONS'])
def tts():
    """
    Génère un WAV depuis du texte.

    Body JSON :
      {
        "text": "Bonjour, ceci est un test.",
        "model": "fr_FR-upmc-medium",
        "speaker_id": 0
      }
    """
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get('text') or '').strip()
    model = data.get('model') or 'fr_FR-upmc-medium'
    speaker_id = int(data.get('speaker_id') or 0)

    if not text:
        return jsonify({'error': 'text vide'}), 400
    if model not in voices:
        return jsonify({
            'error': f'modèle "{model}" non chargé',
            'available': list(voices.keys()),
        }), 404

    voice = voices[model]
    try:
        buf = io.BytesIO()
        syn_cfg = SynthesisConfig(speaker_id=speaker_id) if speaker_id else None
        with wave.open(buf, 'wb') as wav:
            # synthesize_wav configure auto. les paramètres WAV (channels, rate)
            voice.synthesize_wav(text, wav, syn_config=syn_cfg, set_wav_format=True)
        buf.seek(0)
        return send_file(
            buf,
            mimetype='audio/wav',
            as_attachment=False,
            download_name='piper.wav',
        )
    except Exception as e:
        print(f"✗ Erreur synthèse : {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/shutdown', methods=['POST', 'OPTIONS'])
def shutdown():
    """Arrêt propre demandé depuis l'app web."""
    if request.method == 'OPTIONS':
        return ('', 204)
    print("\n⏹ Arrêt demandé depuis l'app web. Bye !")
    # Réponse envoyée AVANT le kill, sinon le client voit une déconnexion
    import threading, time
    def delayed_exit():
        time.sleep(0.3)
        os._exit(0)
    threading.Thread(target=delayed_exit, daemon=True).start()
    return jsonify({'status': 'shutting down'})


@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'Piper TTS local server (ÉpubSon)',
        'endpoints': ['/health (GET)', '/tts (POST)', '/shutdown (POST)'],
        'voices_loaded': list(voices.keys()),
    })


if __name__ == '__main__':
    print("─" * 60)
    print(" Piper TTS local server pour ÉpubSon")
    print("─" * 60)
    load_voices()
    if not voices:
        print()
        print("⚠ Aucune voix chargée. Vérifie le dossier models/.")
        print("  Voir la doc en haut de ce fichier.")
        print()
    print(f"\n✓ Serveur en écoute sur http://{HOST}:{PORT}\n")
    print("  Ouvre epub-to-audiobook.html dans ton navigateur.")
    print("  Ctrl+C pour arrêter.\n")
    # debug=False pour éviter le double load_voices() en mode debug
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
