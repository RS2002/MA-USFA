@echo off
setlocal
cd /d "%~dp0"
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" run.py --stage all %*
exit /b %errorlevel%
