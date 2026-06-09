#!/usr/bin/env python3
"""
Serveur Kokoro TTS local pour ÉpubSon — alternative légère et stable.

────────────────────────────────────────────────────────────────────────
INSTALLATION (une seule fois)
────────────────────────────────────────────────────────────────────────

  1. Python 3.9+ (déjà OK si tu as Piper/XTTS)

  2. Installer les dépendances :
       pip install kokoro-onnx flask flask-cors soundfile

     (Kokoro tourne sur ONNX runtime → CPU-friendly, déterministe,
      pas de bug PyTorch random comme XTTS.)

  3. Télécharger les 2 fichiers du modèle dans `models/` :

       a) Modèle ONNX (~310 Mo) :
          https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx

       b) Voices embeddings (~27 Mo) :
          https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

     → Place les deux dans :
       Zaknaf/
         ├── kokoro_server.py
         └── models/
             ├── kokoro-v1.0.onnx
             └── voices-v1.0.bin

────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────

  Démarrer :
       python kokoro_server.py
  ou   double-clic sur start_kokoro.bat

  Le serveur écoute sur http://127.0.0.1:5007.
  Ctrl+C pour arrêter.

────────────────────────────────────────────────────────────────────────
VOIX FR
────────────────────────────────────────────────────────────────────────

  Kokoro v1.0 contient une voix française native :
    - ff_siwis  : féminine, claire, dataset Siwis (~10h FR)

  Toutes les voix EN/etc sont aussi dispos si tu veux multilingue,
  mais ce serveur filtre sur les voix FR par défaut.

────────────────────────────────────────────────────────────────────────
PERFORMANCE
────────────────────────────────────────────────────────────────────────

  Modèle 82M params + ONNX runtime CPU :
    ~3-5x temps réel sur CPU moderne (1 min audio en 12-20 s)
    Bien plus rapide qu'XTTS-v2 (10-30x), pas de risque OOM.
"""
import gc
import io
import os
import re
import sys
import wave

# UTF-8 sur la console Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# ── Configuration ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
HOST = '127.0.0.1'
PORT = 5007
MODEL_PATH = os.path.join(MODELS_DIR, 'kokoro-v1.0.onnx')
VOICES_PATH = os.path.join(MODELS_DIR, 'voices-v1.0.bin')

# Voix FR à exposer (les autres langues sont ignorées par défaut)
# Préfixe codes Kokoro :
#   ff_* = français féminin   fm_* = français masculin  (s'il y en a)
#   af_* = anglais US féminin am_* = ...
# Pour autoriser plus de voix (anglais etc.), édite cette liste.
ALLOWED_PREFIXES = ('ff_', 'fm_')

# Limite par requête. Kokoro tokenize en G2P → la limite est en tokens,
# pas en chars. ~500 chars c'est large. Si besoin l'app split déjà côté client.
MAX_TEXT_LEN = 800

# ── Imports lourds ────────────────────────────────────────────────────
try:
    from flask import Flask, request, send_file, jsonify
    from flask_cors import CORS
except ImportError:
    print("✗ Flask manquant : pip install flask flask-cors")
    sys.exit(1)

try:
    from kokoro_onnx import Kokoro
except ImportError:
    print("✗ kokoro-onnx manquant. Lance :")
    print("    pip install kokoro-onnx flask flask-cors soundfile")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("✗ numpy manquant (devrait venir avec kokoro-onnx)")
    sys.exit(1)


# ── Vérif des fichiers modèle ────────────────────────────────────────
print("─" * 60)
print(" Kokoro TTS server pour ÉpubSon")
print("─" * 60)

if not os.path.isfile(MODEL_PATH):
    print(f"\n✗ Modèle introuvable : {MODEL_PATH}")
    print(f"\n  Télécharge :")
    print(f"    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
    print(f"  → place-le dans le dossier 'models/'.")
    sys.exit(1)

if not os.path.isfile(VOICES_PATH):
    print(f"\n✗ Voices file introuvable : {VOICES_PATH}")
    print(f"\n  Télécharge :")
    print(f"    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
    print(f"  → place-le dans le dossier 'models/'.")
    sys.exit(1)

print(f"  Modèle  : {os.path.basename(MODEL_PATH)} ({os.path.getsize(MODEL_PATH) // (1024*1024)} Mo)")
print(f"  Voices  : {os.path.basename(VOICES_PATH)} ({os.path.getsize(VOICES_PATH) // (1024*1024)} Mo)")
print(f"  Chargement en cours…", flush=True)

try:
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
except Exception as e:
    print(f"\n✗ Erreur de chargement : {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def list_available_voices():
    """Retourne la liste des voix FR (filtre sur ALLOWED_PREFIXES)."""
    try:
        all_voices = kokoro.get_voices()
    except Exception:
        all_voices = []
    out = []
    for v in sorted(all_voices):
        if v.startswith(ALLOWED_PREFIXES):
            gender = 'F' if v.startswith('ff_') else 'M'
            out.append({'name': v, 'gender': gender, 'language': 'fr'})
    return out


voices = list_available_voices()
print(f"  ✓ Kokoro prêt — {len(voices)} voix FR détectée(s)")
for v in voices:
    print(f"     • {v['name']} ({v['gender']})")

if not voices:
    print(f"\n⚠ Aucune voix FR trouvée. La voix 'ff_siwis' devrait exister.")
    print(f"  Vérifie que le fichier voices-v1.0.bin est à jour.")


# ── Init Flask + CORS ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def add_pna_header(response):
    # Chrome 117+ Private Network Access
    response.headers['Access-Control-Allow-Private-Network'] = 'true'
    return response


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'engine': 'kokoro-onnx',
        'voices': list_available_voices(),
    })


@app.route('/voices', methods=['GET'])
def voices_endpoint():
    return jsonify({'voices': list_available_voices()})


@app.route('/tts', methods=['POST', 'OPTIONS'])
def synth():
    if request.method == 'OPTIONS':
        return ('', 204)

    data = request.get_json(force=True, silent=True) or {}
    text = (data.get('text') or '').strip()
    voice = data.get('voice') or 'ff_siwis'
    speed = float(data.get('speed', 1.0))
    lang = data.get('language', 'fr-fr')

    if not text:
        return jsonify({'error': 'text vide'}), 400

    if len(text) > MAX_TEXT_LEN:
        return jsonify({
            'error': f'texte trop long ({len(text)} > {MAX_TEXT_LEN})',
            'hint': 'split avant envoi',
        }), 413

    # Garde-fou voix
    available = [v['name'] for v in list_available_voices()]
    if voice not in available:
        # Fallback sur la 1ère voix FR dispo
        if available:
            voice = available[0]
        else:
            return jsonify({'error': 'aucune voix FR disponible'}), 503

    try:
        samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        gc.collect()

        # samples = np.array float32 [-1, 1]
        audio_i16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(sr))
            wav.writeframes(audio_i16.tobytes())
        buf.seek(0)
        return send_file(buf, mimetype='audio/wav',
                         as_attachment=False, download_name='kokoro.wav')

    except Exception as e:
        gc.collect()
        print(f"  ✗ Erreur synthèse : {e}")
        print(f"     text='{text[:120]}...'  voice={voice}  lang={lang}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/shutdown', methods=['POST', 'OPTIONS'])
def shutdown():
    if request.method == 'OPTIONS':
        return ('', 204)
    print("\n⏹ Arrêt demandé. Bye !")
    import threading
    import time

    def delayed_exit():
        time.sleep(0.3)
        os._exit(0)

    threading.Thread(target=delayed_exit, daemon=True).start()
    return jsonify({'status': 'shutting down'})


@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'Kokoro TTS server (ÉpubSon)',
        'endpoints': ['/health', '/voices', '/tts', '/shutdown'],
        'voices_count': len(list_available_voices()),
    })


if __name__ == '__main__':
    print()
    print(f"✓ Serveur en écoute sur http://{HOST}:{PORT}\n")
    print(f"  Ctrl+C pour arrêter.\n")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
