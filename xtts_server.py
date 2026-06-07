#!/usr/bin/env python3
"""
Serveur XTTS-v2 (Coqui) local pour EpubSon — voice cloning multilingue.

────────────────────────────────────────────────────────────────────────
INSTALLATION (une seule fois) — séquence VALIDÉE le 2026-06-07
────────────────────────────────────────────────────────────────────────

  Le package 'TTS' original (Coqui) N'EST PLUS MAINTENU et n'est pas
  compatible avec transformers/numpy modernes. Utilise 'coqui-tts' (fork).

  Python 3.9-3.11 (3.10 testé OK) — pas 3.12+

  Séquence d'installation qui MARCHE (Windows CPU) :

      pip install flask flask-cors
      pip install coqui-tts                  (sans [codec], voir note)
      pip install "transformers>=4.57,<5"
      pip install "torch<2.9" "torchaudio<2.9"
      pip install --upgrade pandas
      pip uninstall -y torchcodec            (si présent)

  Pourquoi ces étapes :
   - transformers <5 : le 5.x supprime 'BeamSearchScorer' utilisé par XTTS
   - torch <2.9      : PyTorch 2.9+ exigent torchcodec qui requiert FFmpeg
                       "full-shared" (DLLs séparées) sur Windows. Chocolatey
                       installe la version "essentials" qui ne suffit pas.
                       → bypass : rester sur torch 2.8.x sans torchcodec.
   - pandas >= 2.x   : numpy 2.x est incompatible avec pandas 1.x

  Avec PyTorch 2.8, coqui-tts utilise soundfile/librosa pour l'IO audio
  → pas besoin de torchcodec / FFmpeg full-shared.

  Si l'installation a foiré (anciens TTS résiduels) :
      pip uninstall -y TTS coqui-tts
      rm -rf <python>/Lib/site-packages/TTS
      pip install --force-reinstall coqui-tts[codec]
      pip install --force-reinstall coqpit-config

  Le modèle XTTS-v2 (~2 Go) est téléchargé automatiquement au PREMIER
  lancement vers :
      Windows : %USERPROFILE%/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2/
  Compte 3-5 min de download.

  License CPML pré-acceptée via COQUI_TOS_AGREED=1 ci-dessous.

────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────

  Démarrer (à chaque session) :
       python xtts_server.py

  Ou double-clic sur start_xtts.bat

  Le serveur écoute sur http://127.0.0.1:5006.
  Ctrl+C pour arrêter.

────────────────────────────────────────────────────────────────────────
VOIX CLONÉES
────────────────────────────────────────────────────────────────────────

  Place des fichiers audio (WAV ou MP3) de référence dans le dossier :
       voices/
         ├── pierre.wav     (30s de ta voix)
         ├── narrateur.mp3  (extrait podcast favori)
         └── ...

  XTTS clone la voix à partir de 6+ secondes de sample.
  Recommandation : 20-30 secondes d'audio propre (pas de bruit).

────────────────────────────────────────────────────────────────────────
PERFORMANCE
────────────────────────────────────────────────────────────────────────

  - GPU NVIDIA (CUDA)        : ~1-2x temps réel (rapide)
  - CPU moderne (8 coeurs)   : ~10-30x temps réel (lent mais OK)
  - 1 minute audio générée en : 1-2 min (GPU) / 10-30 min (CPU)
"""
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

# License Coqui CPML pré-acceptée (sinon prompt interactif au 1er run)
os.environ['COQUI_TOS_AGREED'] = '1'

# ── Configuration ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(SCRIPT_DIR, 'voices')
HOST = '127.0.0.1'
PORT = 5006
MODEL_NAME = 'tts_models/multilingual/multi-dataset/xtts_v2'

os.makedirs(VOICES_DIR, exist_ok=True)

# ── Imports lourds ────────────────────────────────────────────────────
try:
    from flask import Flask, request, send_file, jsonify
    from flask_cors import CORS
except ImportError:
    print("✗ Flask manquant : pip install flask flask-cors")
    sys.exit(1)

try:
    from TTS.api import TTS
except ImportError:
    try:
        # Coqui a renommé le package en fin 2024
        from coqui_tts.api import TTS  # noqa
    except ImportError:
        print("✗ TTS manquant. Lance :")
        print("    pip install TTS")
        print("  OU si erreur : pip install coqui-tts")
        sys.exit(1)

try:
    import torch
    import numpy as np
except ImportError:
    print("✗ torch/numpy manquants (devraient être installés avec TTS)")
    sys.exit(1)


def sanitize_name(name: str) -> str:
    """Garde uniquement [a-zA-Z0-9_-] pour éviter path traversal."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', name or '').strip()[:64]


def find_voice_path(name: str):
    """Cherche voices/<name>.wav puis voices/<name>.mp3."""
    name = sanitize_name(name)
    if not name:
        return None
    for ext in ('.wav', '.mp3', '.flac', '.ogg'):
        p = os.path.join(VOICES_DIR, f'{name}{ext}')
        if os.path.isfile(p):
            return p
    return None


def list_voices():
    """Liste tous les fichiers audio dans voices/."""
    out = []
    if not os.path.isdir(VOICES_DIR):
        return out
    for f in sorted(os.listdir(VOICES_DIR)):
        ext = os.path.splitext(f)[1].lower()
        if ext in ('.wav', '.mp3', '.flac', '.ogg'):
            full = os.path.join(VOICES_DIR, f)
            out.append({
                'name': os.path.splitext(f)[0],
                'file': f,
                'size_mb': round(os.path.getsize(full) / (1024 * 1024), 2),
            })
    return out


# ── Chargement du modèle XTTS-v2 ──────────────────────────────────────
print("─" * 60)
print(" XTTS-v2 server pour EpubSon")
print("─" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device : {device.upper()}")
if device == "cpu":
    print(f"  ⚠ Pas de GPU CUDA détecté → génération ~10-30x temps réel.")
    print(f"  ⚠ Pour usage régulier, une GPU NVIDIA est recommandée.")

print(f"  Modèle : {MODEL_NAME}")
print(f"  (téléchargement ~2 Go au premier lancement)")
print(f"  Chargement en cours…", flush=True)

try:
    tts = TTS(MODEL_NAME).to(device)
    print(f"  ✓ XTTS-v2 prêt sur {device.upper()}")
except Exception as e:
    print(f"  ✗ Erreur de chargement : {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Sample rate de sortie XTTS = 24000 Hz
XTTS_SR = 24000


# ── Init Flask + CORS ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'device': device,
        'model': MODEL_NAME,
        'voices': list_voices(),
        'voices_dir': VOICES_DIR,
    })


@app.route('/voices', methods=['GET'])
def voices_list():
    return jsonify({'voices': list_voices()})


@app.route('/upload_voice', methods=['POST', 'OPTIONS'])
def upload_voice():
    if request.method == 'OPTIONS':
        return ('', 204)
    if 'file' not in request.files:
        return jsonify({'error': 'champ file manquant'}), 400
    f = request.files['file']
    raw_name = request.form.get('name') or os.path.splitext(f.filename or '')[0]
    name = sanitize_name(raw_name)
    if not name:
        return jsonify({'error': 'nom invalide (alphanum + _ - uniquement)'}), 400
    # Détermine extension d'origine pour préserver le format
    orig_ext = os.path.splitext(f.filename or '')[1].lower()
    if orig_ext not in ('.wav', '.mp3', '.flac', '.ogg'):
        orig_ext = '.wav'
    save_path = os.path.join(VOICES_DIR, f'{name}{orig_ext}')
    f.save(save_path)
    print(f"  ↑ Voix uploadée : {name}{orig_ext}")
    return jsonify({'status': 'ok', 'name': name, 'file': f'{name}{orig_ext}'})


@app.route('/delete_voice', methods=['POST', 'OPTIONS'])
def delete_voice():
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(force=True, silent=True) or {}
    name = sanitize_name(data.get('name', ''))
    if not name:
        return jsonify({'error': 'nom invalide'}), 400
    deleted = False
    for ext in ('.wav', '.mp3', '.flac', '.ogg'):
        p = os.path.join(VOICES_DIR, f'{name}{ext}')
        if os.path.isfile(p):
            os.remove(p)
            deleted = True
    if not deleted:
        return jsonify({'error': 'voix introuvable'}), 404
    print(f"  🗑 Voix supprimée : {name}")
    return jsonify({'status': 'ok'})


@app.route('/tts', methods=['POST', 'OPTIONS'])
def synth():
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get('text') or '').strip()
    voice = data.get('voice') or ''
    language = (data.get('language') or 'fr').strip().lower()

    # Paramètres XTTS-v2 (defaults orientés naturel)
    speed = float(data.get('speed', 1.0))   # 0.7 = lent expressif, 1.3 = rapide

    if not text:
        return jsonify({'error': 'text vide'}), 400

    voice_path = find_voice_path(voice)
    if not voice_path:
        return jsonify({
            'error': f'voix "{voice}" introuvable',
            'available': [v['name'] for v in list_voices()],
        }), 404

    try:
        # split_sentences=True : XTTS découpe sur la ponctuation → meilleure prosodie
        audio = tts.tts(
            text=text,
            speaker_wav=voice_path,
            language=language,
            speed=speed,
            split_sentences=True,
        )
        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
        # Normalise et convertit en int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(XTTS_SR)
            wav.writeframes(audio_i16.tobytes())
        buf.seek(0)
        return send_file(buf, mimetype='audio/wav',
                         as_attachment=False, download_name='xtts.wav')
    except Exception as e:
        print(f"  ✗ Erreur synthèse : {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/shutdown', methods=['POST', 'OPTIONS'])
def shutdown():
    if request.method == 'OPTIONS':
        return ('', 204)
    print("\n⏹ Arrêt demandé depuis l'app web. Bye !")
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
        'service': 'XTTS-v2 server (ÉpubSon)',
        'endpoints': [
            '/health', '/voices', '/upload_voice', '/delete_voice',
            '/tts', '/shutdown',
        ],
        'device': device,
        'voices_count': len(list_voices()),
    })


if __name__ == '__main__':
    voices_count = len(list_voices())
    print(f"  Voix dispos : {voices_count}")
    if voices_count == 0:
        print(f"  ⚠ Aucune voix dans {VOICES_DIR}")
        print(f"  Place un fichier .wav/.mp3 (6+ s d'audio) puis recharge.")
        print(f"  Ou upload via l'app web.")
    print()
    print(f"✓ Serveur en écoute sur http://{HOST}:{PORT}\n")
    print(f"  Ctrl+C pour arrêter.\n")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
