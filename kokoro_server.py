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


# ── Mots où espeak FR rate la liaison même avec '-' (bug interne) ──
# On force avant tout par substitution caractère :
#   son  + voyelle  -> sõn   (õ = U+00F5, espeak garde /sɔ̃n/ correctement)
#   moyen + voyelle -> moyenn (perd la nasalisation, mais le /n/ liaison passe)
#   doit + voyelle  -> doitt (gagne le /t/ liaison)
ESPEAK_LIAISON_FIXES = [
    (re.compile(rf'\bson\s+([{V}])', re.IGNORECASE), r'sõn \1'),
    (re.compile(rf'\bmoyen\s+([{V}])', re.IGNORECASE), r'moyenn \1'),
    (re.compile(rf'\bdoit\s+([{V}])', re.IGNORECASE), r'doitt \1'),
]

# ── Mots mal prononcés par espeak FR (nasalisation erronée, etc.) ──
# "chaman" -> nasale finale /ɑ̃/ par défaut chez espeak, alors qu'il faut
# /an/ non-nasal. Respeller en "chamane" force la bonne prononciation.
WORD_PRONUNCIATION_FIXES = [
    (re.compile(r'\bchaman(s?)\b', re.IGNORECASE), r'chamane\1'),
]


def apply_french_liaisons(text):
    # 1) Mots simplement mal prononcés (respelling phonétique)
    for pattern, repl in WORD_PRONUNCIATION_FIXES:
        text = pattern.sub(repl, text)
    # 2) Pré-corrections pour mots où espeak rate la liaison nativement
    for pattern, repl in ESPEAK_LIAISON_FIXES:
        text = pattern.sub(repl, text)
    # 3) Règles génériques (trait d'union pour fusion phonétique)
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

    # Ellipsis : on la convertit en placeholder pour qu'elle survive
    # à la transformation virgule->point en aval (sinon "..." serait
    # collapsé en "."). Restauration à la toute fin.
    ELLIPSIS_PH = '\x02'
    text = text.replace('…', ELLIPSIS_PH)

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

    # ── INTONATION INTERROGATIVE (boost modéré) ───────────────────────
    # '??' amplifie le pitch montant sur '?' final. On utilise PAS '???'
    # car espeak étalerait la courbe interrogative sur toute la phrase,
    # faisant remonter le ton sur les virgules internes (non voulu).
    text = re.sub(r'(?<![?!])\?(?![?!])', '??', text)

    # ── INTONATION DESCENDANTE AVANT LES VIRGULES (max) ───────────────
    # Niveaux espeak FR du plus doux au plus profond :
    #   ','        = montée légère (continuation)
    #   ';'        = chute légère
    #   '.'        = chute déclarative moyenne
    #   '!'        = chute emphatique profonde (+30% vs '.')
    #   '!!'       = chute compoundée (+50% vs '.')
    #   '!!!'      = saturation espeak (+70% vs '.', max audible)
    # Demande utilisateur : amplitude maximale -> '!!!'.
    text = re.sub(r'\s*,\s*', '!!! ', text)
    # Collapse '..' / '. .' éventuels (le placeholder ellipsis n'est PAS
    # affecté car c'est un char \x02, pas un point)
    text = re.sub(r'\.\s*\.+', '.', text)

    # Collapse espaces et trim
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.!?;:])', r'\1', text)

    # Restaure ellipsis (pause dramatique préservée)
    text = text.replace(ELLIPSIS_PH, '...')
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

# ── Pool d'instances Kokoro indépendantes (parallélisme réel) ───────────
# Avant : un seul objet Kokoro + un lock global -> tout était sérialisé,
# même en envoyant des requêtes en parallèle (elles faisaient juste la
# queue derrière le lock). Cause du lock : kokoro_onnx a un état interne
# partagé qui se corrompt sous appels concurrents sur LE MÊME objet
# (mots mélangés entre requêtes voisines). Solution : plusieurs instances
# INDÉPENDANTES (chacune son propre état), une seule requête à la fois
# PAR instance -> plus de risque de mélange, tout en autorisant N requêtes
# simultanées au total.
#
# Chaque instance ONNX utilise par défaut tous les cœurs CPU en intra-op
# threading -> avec plusieurs instances en parallèle ça se marcherait
# dessus. On monkey-patch InferenceSession pour bornez chaque instance à
# (cœurs physiques / nb workers) threads.
NUM_KOKORO_WORKERS = int(os.environ.get('KOKORO_WORKERS', '2'))
_cpu_count = os.cpu_count() or 4
_INTRA_OP_THREADS = max(1, _cpu_count // NUM_KOKORO_WORKERS)

import onnxruntime as _ort
_ORIG_INFERENCE_SESSION = _ort.InferenceSession

def _threaded_inference_session(model_path, providers=None, **kwargs):
    so = _ort.SessionOptions()
    so.intra_op_num_threads = _INTRA_OP_THREADS
    so.inter_op_num_threads = 1
    return _ORIG_INFERENCE_SESSION(model_path, sess_options=so, providers=providers, **kwargs)

_ort.InferenceSession = _threaded_inference_session

print(f"  Chargement de {NUM_KOKORO_WORKERS} instance(s) Kokoro ({_INTRA_OP_THREADS} threads chacune)...", flush=True)

import queue
_kokoro_pool = queue.Queue()
try:
    kokoro_instances = []
    for _i in range(NUM_KOKORO_WORKERS):
        print(f"    instance {_i+1}/{NUM_KOKORO_WORKERS}...", flush=True)
        _inst = Kokoro(MODEL_PATH, VOICES_PATH)
        kokoro_instances.append(_inst)
        _kokoro_pool.put(_inst)
    kokoro = kokoro_instances[0]  # référence conservée pour _build_blends() (embeddings identiques sur toutes les instances)
except Exception as e:
    print(f"\nX Erreur de chargement : {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ── Voix fm_drow (modèle PyTorch fine-tuné séparé, pas le moteur ONNX) ──
# Package self-contained : fm_drow.pth (poids), config.json, voices/fm_drow.pt
DROW_DIR = r"C:\Users\EA_ADM\Documents\claude_ai\drizzt_out\fm_drow_kokoro"
DROW_SAMPLE_RATE = 24000
_drow_pipeline = None
_drow_voice = None

def _load_drow():
    """Charge le pipeline PyTorch fm_drow. Best-effort : une erreur ici ne doit
    pas empêcher le serveur ONNX de démarrer, juste priver la voix fm_drow."""
    global _drow_pipeline, _drow_voice
    if not os.path.isdir(DROW_DIR):
        print(f"  ! fm_drow non trouve ({DROW_DIR}) -- voix ignoree")
        return
    try:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        sys.path.insert(0, DROW_DIR)
        from fm_drow import load as _drow_load
        _drow_pipeline, _drow_voice = _drow_load(device='cpu')
        print("  OK fm_drow charge (modele PyTorch fine-tune)")
    except Exception as e:
        print(f"  ! Erreur chargement fm_drow : {e}")

_load_drow()


# ── Voix custom par mixing d'embeddings ─────────────────────────────
# Kokoro v1.0 n'a pas de voix FR mâle native. On en crée une en mélangeant
# une voix mâle anglaise (timbre/grain masculin) avec ff_siwis (intonation FR).
# Le résultat est une voix FR mâle approximative (léger résidu d'accent
# selon la base mâle source).
#
# Format : { 'nom_publique': (voix_male, voix_fr, ratio_male) }
# Ratio_male = 0.6 = 60% timbre mâle / 40% inflexion FR Siwis.
CUSTOM_BLENDS = {
    'fm_george': ('bm_george', 'ff_siwis', 0.50),
}

# Cache des embeddings blendés (calculé au démarrage)
__blend_cache = {}


def _build_blends():
    """Pré-calcule les embeddings mélangés au démarrage."""
    global __blend_cache
    for name, (male_v, fr_v, ratio) in CUSTOM_BLENDS.items():
        try:
            male = kokoro.voices[male_v]
            fr = kokoro.voices[fr_v]
            blend = ratio * male + (1.0 - ratio) * fr
            __blend_cache[name] = blend
            print(f"  OK blend '{name}' = {ratio:.0%} {male_v} + {1-ratio:.0%} {fr_v}")
        except Exception as e:
            print(f"  ! blend '{name}' echec : {e}")


_build_blends()


def list_available_voices():
    """Retourne la liste des voix FR (natives + custom blends)."""
    try:
        all_voices = kokoro.get_voices()
    except Exception:
        all_voices = []
    out = []
    for v in sorted(all_voices):
        if v.startswith(ALLOWED_PREFIXES):
            gender = 'F' if v.startswith('ff_') else 'M'
            out.append({'name': v, 'gender': gender, 'language': 'fr', 'custom': False})
    # Ajoute les blends custom
    for name in sorted(__blend_cache.keys()):
        if not name.startswith(ALLOWED_PREFIXES):
            continue
        gender = 'F' if name.startswith('ff_') else 'M'
        out.append({'name': name, 'gender': gender, 'language': 'fr', 'custom': True})
    # Voix fm_drow (modèle fine-tuné à part, pas un blend)
    if _drow_pipeline is not None:
        out.append({'name': 'fm_drow', 'gender': 'M', 'language': 'fr', 'custom': True})
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

    Voix ONNX (ff_siwis, fm_george, ...) : empruntée au pool de N instances
    Kokoro indépendantes -- une seule requête à la fois PAR instance (donc
    pas de risque de mélange), mais jusqu'à N requêtes en parallèle au total.

    fm_drow reste sérialisée via _SYNTH_LOCK (moteur PyTorch séparé, pas
    encore validé en multi-instance).

    Si 'voice' est le nom d'un blend custom, on passe l'embedding numpy
    directement au lieu d'un nom de voix.
    """
    # fm_drow : moteur PyTorch séparé, pas le moteur ONNX kokoro_onnx
    if voice == 'fm_drow' and _drow_pipeline is not None:
        with _SYNTH_LOCK:
            chunks = [a for _gs, _ps, a in _drow_pipeline(text, voice=_drow_voice, speed=speed)]
        if not chunks:
            raise RuntimeError('Synthèse fm_drow vide')
        return np.concatenate(chunks).astype('float32'), DROW_SAMPLE_RATE

    # Voix custom (blend pré-calculé) -> passe l'embedding numpy
    voice_arg = __blend_cache.get(voice, voice)
    inst = _kokoro_pool.get()
    try:
        samples, sr = inst.create(text, voice=voice_arg, speed=speed, lang=lang)
    finally:
        _kokoro_pool.put(inst)
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

    if voice == 'fm_drow':
        # fm_drow a son propre G2P (misaki/espeak) entraîné sur du texte FR
        # normal : les hacks de liaison (virgule -> '!!!', trait d'union) sont
        # spécifiques au moteur ONNX/espeak et dégradent fm_drow. Mais les
        # corrections de prononciation (respelling) restent valables : misaki
        # passe aussi par espeak-ng en interne, donc les mêmes mots posent
        # les mêmes problèmes de nasalisation.
        text = re.sub(r'\s+', ' ', raw_text).strip()
        for pattern, repl in WORD_PRONUNCIATION_FIXES:
            text = pattern.sub(repl, text)
    else:
        text = sanitize_text(raw_text)
        if apply_liaisons and lang.startswith('fr'):
            text = apply_french_liaisons(text)

    if len(text) > MAX_TEXT_LEN:
        return jsonify({
            'error': f'texte trop long ({len(text)} > {MAX_TEXT_LEN})',
            'hint': 'split avant envoi',
        }), 413

    available = [v['name'] for v in list_available_voices()]
    # Voix valide = soit native dans le bin Kokoro, soit dans le blend cache
    if voice not in available and voice not in __blend_cache:
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
