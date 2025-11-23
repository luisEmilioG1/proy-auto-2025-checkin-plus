@echo off
echo ================================================
echo Sistema de Reconocimiento Facial
echo ================================================
echo.
echo Selecciona una opcion:
echo 1. Entrenar modelo desde video
echo 2. Reconocimiento en tiempo real
echo 3. Salir
echo.
set /p choice="Opcion: "

if "%choice%"=="1" goto train
if "%choice%"=="2" goto recognize
if "%choice%"=="3" goto end
goto menu

:train
echo.
echo === ENTRENAR MODELO ===
echo.
set /p video="Ruta del video: "
set /p persons="Nombres de personas (separados por espacios): "
set /p model="Nombre del modelo (Enter para face_model.xml): "
if "%model%"=="" set model=face_model.xml
echo.
echo Ejecutando entrenamiento...
python train_model.py "%video%" --persons %persons% --model %model%
pause
goto end

:recognize
echo.
echo === RECONOCIMIENTO EN TIEMPO REAL ===
echo.
set /p model="Ruta del modelo (Enter para face_model.xml): "
if "%model%"=="" set model=face_model.xml
echo.
echo Iniciando reconocimiento...
python recognize.py --model "%model%"
pause
goto end

:end
exit

