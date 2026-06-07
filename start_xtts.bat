@echo off
REM ====================================================================
REM  Lanceur du serveur XTTS-v2 (voice cloning) pour EpubSon
REM  Double-clique sur ce fichier pour demarrer.
REM  Premier run : telechargement du modele ~2 Go.
REM ====================================================================
chcp 65001 >nul
cd /d "%~dp0"
title XTTS-v2 server
echo.
echo  ================================================
echo   Demarrage XTTS-v2 (voice cloning)
echo   Premier run : telechargement modele ~2 Go
echo   CPU : lent (10-30x temps reel)
echo   GPU NVIDIA : rapide (1-2x temps reel)
echo  ================================================
echo.
python xtts_server.py
echo.
echo  ================================================
echo   Serveur arrete.
echo  ================================================
pause
