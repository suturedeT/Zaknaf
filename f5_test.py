#!/usr/bin/env python3
"""Test basique de F5-TTS avec le modèle français RASPIAUDIO."""
import sys, io, os, time

# Contourne SSL Windows : utilise le cert store du système
# (pip install pip_system_certs aurait fait ça transparemment)
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch requests.adapters.HTTPAdapter pour disable verify globalement
# sans casser les sous-classes (contrairement au patch de Session)
import requests.sessions
_orig_request = requests.sessions.Session.request
def _patched_request(self, method, url, *args, **kwargs):
    kwargs['verify'] = False
    return _orig_request(self, method, url, *args, **kwargs)
requests.sessions.Session.request = _patched_request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Chargement F5-TTS API...", flush=True)
t0 = time.time()
from f5_tts.api import F5TTS

MODELS = r'C:\Users\EA_ADM\Documents\Zaknaf\models\f5_french'
print(f"Init F5TTS avec modèle FR (peut prendre 1-2 min première fois)...", flush=True)

f5 = F5TTS(
    model='F5TTS_Base',  # architecture base (compatible avec le fine-tune RASPIAUDIO)
    ckpt_file=os.path.join(MODELS, 'model_last_reduced.pt'),
    vocab_file=os.path.join(MODELS, 'vocab.txt'),
    device='cpu',
)
print(f"  OK F5TTS prêt en {time.time()-t0:.0f}s", flush=True)

# Sample test : utilise un fichier WAV qu'on a (test_tom comme référence)
# Plus tard on prendra un vrai sample LibriVox
ref_audio = r'C:\Users\EA_ADM\Documents\Zaknaf\test_tom.wav'
ref_text = "Bonjour, je suis Tom, narrateur français. Le doyen des prêtres descendit lentement vers la cité, perdu dans ses pensées."

gen_text = "Bonjour. Ceci est un test de la voix clonée par F5-TTS sur le modèle français RASPIAUDIO."

print(f"Génération test...", flush=True)
t0 = time.time()
wav_path = r'C:\Users\EA_ADM\Documents\Zaknaf\test_f5_first.wav'
f5.infer(
    ref_file=ref_audio,
    ref_text=ref_text,
    gen_text=gen_text,
    file_wave=wav_path,
)
print(f"  OK généré en {time.time()-t0:.0f}s : {wav_path}", flush=True)
