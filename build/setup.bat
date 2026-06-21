@echo off
REM Music-Assassin Build System Setup
REM Checks prerequisites and prepares build environment

echo.
echo ============================================================
echo   Music-Assassin Build System Setup
echo ============================================================
echo.

echo Checking prerequisites...

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python not found (install Python 3.8+ from python.org)
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_ver=%%i
    echo [OK] Python !python_ver! found
)

REM Check FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [X] FFmpeg not found (install from https://ffmpeg.org)
) else (
    echo [OK] FFmpeg found
)

REM Check PyInstaller
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [!] PyInstaller not installed
    echo     Run: pip install -r requirements_packaging.txt
) else (
    echo [OK] PyInstaller installed
)

echo.
echo ============================================================
echo   Build System Ready!
echo ============================================================
echo.
echo Available versions:
echo   build\build.bat 0.0.1   - Initial MoviePy version
echo   build\build.bat 0.0.2   - MoviePy with Logger
echo   build\build.bat 0.1.0   - FFMPEG Transition
echo   build\build.bat 0.2.0   - FFMPEG Cleanup ^& Optimization
echo   build\build.bat 0.3.3   - Enhanced Edition ^(Harmonic Rejection^)
echo   build\build.bat 0.4.1   - DEMUCS/Spleeter Hybrid ^(Current^)
echo.
echo Example:
echo   build\build.bat 0.4.1
echo.
echo For more information:
echo   type build\README.md
echo.
pause
