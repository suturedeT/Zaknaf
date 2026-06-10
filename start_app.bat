@echo off
REM ====================================================================
REM  Lanceur du serveur HTTP local pour l'app EpubSon
REM
REM  Pourquoi : Chrome 117+ bloque les requetes d'un origin HTTPS public
REM  (github.io) vers loopback (127.0.0.1) via la policy PNA. En servant
REM  l'app depuis localhost, on reste dans le meme espace -> aucun blocage.
REM
REM  Double-clique sur ce fichier pour demarrer. La page s'ouvre dans
REM  ton navigateur par defaut. Garde le terminal ouvert tant que tu
REM  utilises l'app. Ctrl+C pour arreter.
REM ====================================================================
chcp 65001 >nul
cd /d "%~dp0"
title EpubSon - serveur app local

echo.
echo  ================================================
echo   Serveur app EpubSon sur http://localhost:8000
echo  ================================================
echo.
echo   Ouvre dans Chrome :
echo     http://localhost:8000/epub-to-audiobook.html
echo.
echo   Ne ferme pas cette fenetre tant que tu utilises l'app.
echo   Ctrl+C pour arreter.
echo.
echo  ================================================
echo.

REM Lance le navigateur par defaut sur l'app (apres 1 sec pour laisser
REM le serveur demarrer)
start "" cmd /c "timeout /t 1 >nul && start http://localhost:8000/epub-to-audiobook.html"

REM Demarre le serveur HTTP Python (bloquant)
python -m http.server 8000

echo.
echo  ================================================
echo   Serveur arrete.
echo  ================================================
pause
