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

     -> Place les deux dans :
       Zaknaf/
         |-- kokoro_server.py
         `-- models/
             |-- kokoro-v1.0.onnx
             `-- voices-v1.0.bin

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

────────────────────────────────────────────────────────────────────────
"""
import gc
import io
import os
import re
import sys
import threading
import wave

# Lock global : kokoro_onnx.Kokoro a un état interne (session ONNX + tokenizer
# + tampons) qui peut être corrompu par appels concurrents Flask threaded.
# Symptôme observé : audio de la requête A retourné à la requête B, mots
# mélangés entre phrases adjacentes. La synthèse étant courte (~secondes),
# sérialiser n'a pas d'impact perceptible sur le throughput global.
_SYNTH_LOCK = threading.Lock()

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
ALLOWED_PREFIXES = ('ff_', 'fm_')

# Limite par requête (chars source, avant phonemization)
MAX_TEXT_LEN = 800

# ── Imports lourds ────────────────────────────────────────────────────
try:
    from flask import Flask, request, send_file, jsonify
    from flask_cors import CORS
except ImportError:
    print("X Flask manquant : pip install flask flask-cors")
    sys.exit(1)

try:
    from kokoro_onnx import Kokoro
except ImportError:
    print("X kokoro-onnx manquant. Lance :")
    print("    pip install kokoro-onnx flask flask-cors soundfile")
    sys.exit(1)

# ─── Monkey-patch phonemizer : désactive le words_mismatch ───────────
# Bug 1 (HTTP 500) : _mismatched_lines() raise quand espeak produit un
# nombre de lignes différent de l'input. Cause : nom propre exotique
# (Yozumian) que espeak décompose en syllabes -> compte de "lignes" différent.
# Bug 2 (mots droppés) : Remove.process() supprime carrément les lignes
# mismatch -> "Car le doyen des prêtres" devient "c pretre" si le bloc
# entier est jeté.
# Solution : on remplace _mismatched_lines() pour ne JAMAIS raise et
# retourner liste vide -> aucun drop, juste les phonèmes bruts d'espeak.
try:
    from phonemizer.backend.espeak import words_mismatch as _wm_mod
    _wm_mod.BaseWordsMismatch._mismatched_lines = lambda self: []
    _wm_mod.BaseWordsMismatch._resume = lambda self, n, m: None
    # Override aussi les classes concrètes pour être sûr
    _wm_mod.Ignore.process = lambda self, text: text
    _wm_mod.Warn.process = lambda self, text: text
    _wm_mod.Remove.process = lambda self, text: text
    print("  OK phonemizer monkey-patche (words_mismatch desactive)")
except Exception as _e:
    print(f"  ! Patch phonemizer impossible : {_e}")
    print(f"    (le bug 'mots droppes' peut subsister)")

try:
    import numpy as np
except ImportError:
    print("X numpy manquant (devrait venir avec kokoro-onnx)")
    sys.exit(1)


# ── Liaisons françaises (même logique que piper_server.py) ──────────
# espeak-ng (utilisé par Kokoro via phonemizer) ne fait PAS les liaisons FR
# obligatoires. On insère un trait d'union → espeak fusionne les phonèmes.
V = r'aeiouéèêëàâîïôöùûüœhAEIOUÉÈÊËÀÂÎÏÔÖÙÛÜŒH'
LIAISON_RULES = [
    (re.compile(rf'\b(les|des|mes|tes|ses|ces|nos|vos|leurs|aux|quelques|plusieurs)\s+([{V}])', re.IGNORECASE), r'\1-\2'),
    (re.compile(rf'\b(nous|vous|ils|elles|on)\s+([{V}])', re.IGNORECASE), r'\1-\2'),
    (re.compile(rf'\b(en|y)\s+([{V}])', re.IGNORECASE), r'\1-\2'),
    (re.compile(rf'\b(un|aucun|mon|ton|son|bon|moyen|certain)\s+([{V}])', re.IGNORECASE), r'\1-\2'),
    (re.compile(rf'\b(petit|petits|grand|grands|gros|haut|hauts|tout|tous|saint|vingt|cent|fort|forts|long|longs|premier|premiers|dernier|derniers)\s+([{V}])', re.IGNORECASE), r'\1-\2'),
    (re.compile(rf'\b(chez|sous|sans|dans|dès|pendant|avant|après|sauf|devant)\s+([{V}])', re.IGNORECASE), r'\1-\2'),
    (re.compile(rf'\b(est|sont|était|étaient|sera|seront|fait|fut|peut|veut|doit|prend|tient|vient|sait|dit|met|paraît)\s+([{V}])', re.IGNORECASE), r'\1-\2'),
    (re.compile(rf'\b(mais|puis|donc|quand|trop|fort|bien|comment|combien|moins|plus)\s+([{V}])', re.IGNORECASE), r'\1-\2'),
    (re.compile(rf'\b(ont|avons|avez|sommes|êtes)\s+([{V}])', re.IGNORECASE), r'\1-\2'),
]
H_ASPIRE = {'haricot','héros','hibou','hache','haie','haine','hâte','haut','hauteur',
            'hall','halle','halte','hamac','hamster','hanche','handicap','hangar',
            'harnais','harpe','hasard','hère','hérisson','hisser','hocher','homard',
            'hongrois','honte','hotte','houblon','hublot','huit','hurler','hutte'}

# Caractères Unicode invisibles à supprimer (zero-width, BOM, soft hyphen, marks)
INVISIBLE_CHARS_RE = re.compile(
    '[' + ''.join([
        '­',  # soft hyphen
        '​',  # zero-width space
        '‌',  # ZWNJ
        '‍',  # ZWJ
        '‎',  # LTR mark
        '‏',  # RTL mark
        '‪-‮',  # bidi controls
        '⁠',  # word joiner
        '﻿',  # BOM
    ]) + ']'
)


def apply_french_liaisons(text):
    for pattern, repl in LIAISON_RULES:
        text = pattern.sub(repl, text)
    for h in H_ASPIRE:
        text = re.sub(rf'-({h[0]}{h[1:]})', r' \1', text, flags=re.IGNORECASE)
    return text


def sanitize_text(text):
    """Nettoie le texte pour éviter les bugs phonemizer (input/output mismatch).

    Cause connue : espeak-ng compte différemment les "lignes" entrée/sortie
    quand le texte contient des sauts de ligne, doubles espaces, ou certains
    caractères Unicode. On force un texte mono-ligne propre.
    """
    # Mono-ligne : tout whitespace -> espace simple
    text = re.sub(r'\s+', ' ', text)
    # Normalise apostrophes courbes -> droite (préserve les élisions)
    text = text.replace('‘', "'").replace('’', "'")

    # ── INTONATION : transformation des marqueurs typographiques en
    #    ponctuation prosodique que espeak/Kokoro respectent.
    #
    # Em-dash / en-dash : usage français = incise OU début de dialogue.
    # - Incise "Il sortit — comme prévu — sans bruit" :
    #     virgule = pause courte qui préserve la continuité de phrase.
    # - Dialogue "— Bonjour, dit-il" en début de ligne :
    #     virgule reste correct (légère pause avant la prise de parole).
    # On utilise " , " avec espaces pour que espeak la traite comme
    # ponctuation séparée (pas attachée au mot précédent).
    text = re.sub(r'\s*[—–]\s*', ' , ', text)

    # Guillemets : ouverture/fermeture = pause de dialogue. Remplacer
    # par virgule + espace pour signaler une transition prosodique
    # sans que espeak lise "ouvrir/fermer guillemets".
    text = re.sub(r'\s*[«“]\s*', ' , ', text)  # ouvrants
    text = re.sub(r'\s*[»”]\s*', ' , ', text)  # fermants
    text = text.replace('"', ' , ')

    # Ellipsis : conserver comme "..." (espeak la fait sonner comme
    # une pause dramatique allongée — naturelle pour la narration).
    text = text.replace('…', '...')

    # Supprime caractères invisibles
    text = INVISIBLE_CHARS_RE.sub('', text)

    # Nettoyage : supprime virgules adjacentes (" , , ") créées par
    # juxtaposition de typographies, garde une seule.
    text = re.sub(r'\s*,(\s*,)+\s*', ', ', text)
    # Virgule immédiatement après une ponctuation forte est inutile :
    # ". ," -> ".", "! ," -> "!", "? ," -> "?"
    text = re.sub(r'([.!?])\s*,\s*', r'\1 ', text)
    # Virgule juste avant ponctuation forte : ", ." -> "."
    text = re.sub(r'\s*,\s*([.!?])', r'\1', text)
    # Collapse doubles espaces résiduels
    text = re.sub(r'\s+', ' ', text)
    # Trim espace avant ponctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)

    # ── RENFORCEMENT INTONATION INTERROGATIVE ────────────────────────
    # espeak-ng FR a une intonation montante très faible sur '?' isolé.
    # Chaque '?' supplémentaire amplifie la modulation pitch montante :
    #   ?    = espeak natif (faible)
    #   ??   = ~+50% pitch montant
    #   ???  = ~+70% pitch montant (réglage utilisateur actuel)
    # On n'applique qu'aux phrases interrogatives normales, pas aux '?!'
    # ou '??' déjà présents (sinon répétition incontrôlée).
    text = re.sub(r'(?<![?!])\?(?![?!])', '???', text)

    return text.strip()


def split_into_sentences(text):
    """Découpe sur ponctuation forte pour fallback. Renvoie liste de fragments."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


# ── Vérif des fichiers modèle ────────────────────────────────────────
print("-" * 60)
print(" Kokoro TTS server pour EpubSon")
print("-" * 60)

if not os.path.isfile(MODEL_PATH):
    print(f"\nX Modèle introuvable : {MODEL_PATH}")
    print(f"  Télécharge kokoro-v1.0.onnx dans models/")
    sys.exit(1)

if not os.path.isfile(VOICES_PATH):
    print(f"\nX Voices file introuvable : {VOICES_PATH}")
    print(f"  Télécharge voices-v1.0.bin dans models/")
    sys.exit(1)

print(f"  Modèle  : {os.path.basename(MODEL_PATH)} ({os.path.getsize(MODEL_PATH) // (1024*1024)} Mo)")
print(f"  Voices  : {os.path.basename(VOICES_PATH)} ({os.path.getsize(VOICES_PATH) // (1024*1024)} Mo)")
print(f"  Chargement en cours...", flush=True)

try:
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
except Exception as e:
    print(f"\nX Erreur de chargement : {e}")
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
print(f"  OK Kokoro pret -- {len(voices)} voix FR detectee(s)")
for v in voices:
    print(f"     - {v['name']} ({v['gender']})")


# ── Init Flask + CORS ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def add_pna_header(response):
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


def synth_one(text, voice, speed, lang):
    """Synthétise UN fragment, propage l'exception phonemizer si besoin.

    Sérialisé via _SYNTH_LOCK : appels concurrents corrompent l'état interne
    de kokoro_onnx (audio cross-talk entre requêtes Flask threaded).
    """
    with _SYNTH_LOCK:
        samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)
    return samples, sr


def synth_with_fallback(text, voice, speed, lang):
    """Synthèse robuste : essaye d'un coup, sinon découpe en phrases et concat.

    Le bug phonemizer 'lines mismatch' est aléatoire selon le texte. La parade
    fiable est de découper en phrases plus courtes.
    """
    try:
        return synth_one(text, voice, speed, lang)
    except RuntimeError as e:
        msg = str(e).lower()
        if 'lines in input and output' not in msg and 'mismatch' not in msg:
            raise
        print(f"  ! phonemizer mismatch, fallback split phrases ({len(text)} chars)")

    parts = split_into_sentences(text)
    if len(parts) <= 1:
        mid = len(text) // 2
        space = text.rfind(' ', 0, mid + 30)
        if space > 20:
            parts = [text[:space].strip(), text[space:].strip()]
        else:
            raise RuntimeError("phonemizer mismatch sur texte insplittable")

    audio_chunks = []
    sr = None
    for p in parts:
        if not p:
            continue
        try:
            s, this_sr = synth_one(p, voice, speed, lang)
            audio_chunks.append(s)
            sr = this_sr
        except RuntimeError as e2:
            if 'lines' in str(e2):
                sub_parts = split_into_sentences(p)
                if len(sub_parts) > 1:
                    for sp in sub_parts:
                        try:
                            s, this_sr = synth_one(sp, voice, speed, lang)
                            audio_chunks.append(s)
                            sr = this_sr
                        except Exception:
                            print(f"     X sub-fragment dropped : {sp[:60]!r}")
                else:
                    print(f"     X fragment dropped : {p[:60]!r}")
            else:
                raise

    if not audio_chunks:
        raise RuntimeError("Tous les fragments ont echoue")

    combined = np.concatenate(audio_chunks)
    return combined, sr


@app.route('/tts', methods=['POST', 'OPTIONS'])
def synth():
    if request.method == 'OPTIONS':
        return ('', 204)

    data = request.get_json(force=True, silent=True) or {}
    raw_text = (data.get('text') or '').strip()
    voice = data.get('voice') or 'ff_siwis'
    speed = float(data.get('speed', 1.0))
    lang = data.get('language', 'fr-fr')
    apply_liaisons = data.get('apply_liaisons', True)

    if not raw_text:
        return jsonify({'error': 'text vide'}), 400

    text = sanitize_text(raw_text)

    if apply_liaisons and lang.startswith('fr'):
        text = apply_french_liaisons(text)

    if len(text) > MAX_TEXT_LEN:
        return jsonify({
            'error': f'texte trop long ({len(text)} > {MAX_TEXT_LEN})',
            'hint': 'split avant envoi',
        }), 413

    available = [v['name'] for v in list_available_voices()]
    if voice not in available:
        if available:
            voice = available[0]
        else:
            return jsonify({'error': 'aucune voix FR disponible'}), 503

    try:
        samples, sr = synth_with_fallback(text, voice, speed, lang)
        gc.collect()

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
        print(f"  X Erreur synthese : {e}")
        print(f"     text={text[:150]!r}  voice={voice}  lang={lang}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/shutdown', methods=['POST', 'OPTIONS'])
def shutdown():
    if request.method == 'OPTIONS':
        return ('', 204)
    print("\n[STOP] Arret demande. Bye !")
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
        'service': 'Kokoro TTS server (EpubSon)',
        'endpoints': ['/health', '/voices', '/tts', '/shutdown'],
        'voices_count': len(list_available_voices()),
    })


if __name__ == '__main__':
    print()
    print(f"OK Serveur en ecoute sur http://{HOST}:{PORT}\n")
    print(f"  Ctrl+C pour arreter.\n")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
