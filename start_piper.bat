@echo off
REM ====================================================================
REM  Lanceur du serveur Piper TTS pour EpubSon
REM  Double-clique sur ce fichier pour demarrer le serveur.
REM  Ctrl+C ou ferme la fenetre pour arreter.
REM ====================================================================
chcp 65001 >nul
cd /d "%~dp0"
title Piper TTS server
echo.
echo  ================================================
echo   Demarrage du serveur Piper TTS pour EpubSon...
echo  ================================================
echo.
python piper_server.py
echo.
echo  ================================================
echo   Serveur arrete.
echo  ================================================
pause
