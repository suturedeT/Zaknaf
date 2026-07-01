@echo off
REM ====================================================================
REM  Lanceur "tout-en-un" pour EpubSon
REM  Demarre Piper + XTTS-v2 + Kokoro (fm_drow) chacun dans sa propre
REM  fenetre, puis le serveur app local et ouvre le navigateur dessus.
REM
REM  Pour arreter un service : ferme sa fenetre (ou Ctrl+C dedans).
REM  Pour tout arreter : ferme les 3 fenetres de serveurs TTS puis
REM  Ctrl+C (ou ferme) cette fenetre-ci.
REM ====================================================================
chcp 65001 >nul
cd /d "%~dp0"
title EpubSon - tout demarrer

echo.
echo  ================================================
echo   EpubSon - demarrage de tous les services
echo  ================================================
echo.

echo  [1/4] Piper TTS (port 5005)...
start "Piper TTS - EpubSon" cmd /k "chcp 65001 >nul && python piper_server.py"

echo  [2/4] XTTS-v2 (port 5006)...
start "XTTS-v2 - EpubSon" cmd /k "chcp 65001 >nul && python xtts_server.py"

echo  [3/4] Kokoro fm_drow (port 5007)...
start "Kokoro fm_drow - EpubSon" cmd /k "chcp 65001 >nul && python kokoro_server.py"

echo  [4/4] Serveur app (port 8000)...
echo.
echo  ================================================
echo   3 fenetres de serveurs TTS viennent de s'ouvrir.
echo   Laisse-les ouvertes tant que tu utilises l'app.
echo   (Kokoro et XTTS peuvent prendre 10-30s a charger.)
echo.
echo   Cette fenetre demarre le serveur app puis ouvre
echo   ton navigateur. Ne la ferme pas non plus.
echo  ================================================
echo.

start "" cmd /c "timeout /t 2 >nul && start http://localhost:8000/epub-to-audiobook.html"
python -m http.server 8000

echo.
echo  ================================================
echo   Serveur app arrete.
echo  ================================================
pause
