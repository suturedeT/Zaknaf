@echo off
REM ====================================================================
REM  Lanceur du serveur Kokoro TTS pour EpubSon
REM  Double-clique sur ce fichier pour demarrer.
REM ====================================================================
chcp 65001 >nul
cd /d "%~dp0"
title Kokoro TTS server
echo.
echo  ================================================
echo   Demarrage Kokoro TTS (fine-tune maison)
echo   ~2x temps reel sur CPU, voix FR fm_drow
echo  ================================================
echo.
python kokoro_server.py
echo.
echo  ================================================
echo   Serveur arrete.
echo  ================================================
pause
