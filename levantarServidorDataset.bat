@echo off
setlocal
REM Cambiar al directorio del proyecto
cd /d F:\ApiModels\apiPrPthDataset\ModelosEntrenamiento
REM Verificar que Python existe (ruta fija de tu equipo)
set PYTHON_PATH=C:\Users\neotrading\AppData\Local\Programs\Python\Python313\python.exe
IF NOT EXIST "%PYTHON_PATH%" (
    echo ERROR: No se encontro Python en la ruta configurada.
    pause
    exit /b
)
REM Verificar que el entorno virtual existe
IF NOT EXIST "env\Scripts\activate.bat" (
    echo El entorno virtual no existe. Creandolo...
    "%PYTHON_PATH%" -m venv env
)
REM Activar el entorno virtual
call env\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR al activar el entorno virtual.
    pause
    exit /b
)
REM Mostrar mensaje de inicio
echo Entorno virtual activado.
echo Iniciando servidor...
REM Ejecutar el servidor
"%PYTHON_PATH%" -m uvicorn apiDataset:app --host 172.16.0.4 --port 80 --reload
REM Mostrar resultado
IF %ERRORLEVEL% EQU 0 (
    echo Servidor iniciado correctamente.
) ELSE (
    echo ERROR al iniciar el servidor. Codigo de salida: %ERRORLEVEL%
)
pause