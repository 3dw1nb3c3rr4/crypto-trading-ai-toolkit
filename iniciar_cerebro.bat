@echo off
REM ==== Cerebro: abre el panel y ejecuta el bot del dia (si aun no se ejecuto hoy) ====
REM Doble clic en este archivo, o crea un acceso directo en el escritorio.
REM Si tu entorno virtual esta en otra carpeta, cambia la linea VENV de abajo.
set "VENV=C:\Users\edwin\Downloads\borrar\techosuelo\.venv"
if exist "%~dp0.venv\Scripts\activate.bat" set "VENV=%~dp0.venv"
if defined CEREBRO_VENV set "VENV=%CEREBRO_VENV%"

cd /d "%~dp0"
if not exist "%VENV%\Scripts\activate.bat" (
  echo No encuentro el entorno virtual en "%VENV%". Edita la linea VENV de este archivo.
  pause
  exit /b 1
)
call "%VENV%\Scripts\activate.bat"
echo Abriendo el panel en http://127.0.0.1:8765  (cierra esta ventana o pulsa Ctrl+C para apagarlo)
python bots\xs_daily\dashboard.py --auto-bot
pause
