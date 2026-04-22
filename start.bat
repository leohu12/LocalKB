@echo off
title LocalKB - 本地知识库
echo.
echo  ==============================
echo   LocalKB - Local Knowledge Base
echo   http://localhost:5800
echo  ==============================
echo.

cd /d D:\KnowledgeBase

:: Start server
start "" http://localhost:5800
C:\Users\leo\AppData\Local\Programs\Python\Python311\python.exe app.py

pause
