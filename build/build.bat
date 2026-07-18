@echo off
REM Music-Assassin Build Script - Version Selector
REM Builds any version specified by the user

setlocal enabledelayedexpansion

REM Virtual Environment Path
REM Can be set as environment variable: set VENV_PATH=C:\path\to\venv
REM Or will auto-detect from common locations
if "!VENV_PATH!"=="" (
    if exist "%USERPROFILE%\Documents\venvs\call_venv\Scripts\activate.bat" (
        set "VENV_PATH=%USERPROFILE%\Documents\venvs\call_venv"
    ) else if exist "%USERPROFILE%\Documents\venvs\assassin_venv\Scripts\activate.bat" (
        set "VENV_PATH=%USERPROFILE%\Documents\venvs\assassin_venv"
    ) else if exist "venv\Scripts\activate.bat" (
        set "VENV_PATH=venv"
    ) else if exist ".venv\Scripts\activate.bat" (
        set "VENV_PATH=.venv"
    )
)

echo.
echo ============================================================
echo   Music Assassin - PyInstaller Build Script
echo ============================================================
echo.

if "%1"=="" (
    echo Usage: build.bat [version]
    echo.
    echo Available versions:
    echo   build.bat 0.0.1   - Initial MoviePy version
    echo   build.bat 0.0.2   - MoviePy with Logger
    echo   build.bat 0.1.0   - FFMPEG Transition
    echo   build.bat 0.2.0   - FFMPEG Cleanup ^& Optimization
    echo   build.bat 0.3.3   - Enhanced Edition (Harmonic Rejection)
    echo   build.bat 0.3.4   - YouTube Edition
    echo   build.bat 0.4.0   - DEMUCS Natural Vocals
    echo   build.bat 0.4.1   - DEMUCS/Spleeter Hybrid
    echo   build.bat 0.4.2   - DEMUCS/Spleeter Hybrid (memory-optimised)
    echo   build.bat 0.5.0   - CPU Streaming Edition (Current)
    echo.
    echo Example:
    echo   build.bat 0.5.0
    echo.
    pause
    exit /b 1
)

set VERSION=%1
set EXECUTABLE_FILE=versions\MusicAssassinGUI_v%VERSION%.py
set SPEC_FILE=build\MusicAssassin_v%VERSION%.spec
set BUILD_DIR=build\output_v%VERSION%

REM Validate version
if not "%VERSION%"=="0.0.1" if not "%VERSION%"=="0.0.2" if not "%VERSION%"=="0.1.0" if not "%VERSION%"=="0.2.0" if not "%VERSION%"=="0.3.3" if not "%VERSION%"=="0.3.4" if not "%VERSION%"=="0.4.0" if not "%VERSION%"=="0.4.1" if not "%VERSION%"=="0.4.2" if not "%VERSION%"=="0.5.0" (
    echo ERROR: Invalid version '%VERSION%'
    echo Valid versions: 0.0.1, 0.0.2, 0.1.0, 0.2.0, 0.3.3, 0.3.4, 0.4.0, 0.4.1, 0.4.2, 0.5.0
    pause
    exit /b 1
)

REM Check if files exist
if not exist "%EXECUTABLE_FILE%" (
    echo ERROR: Python file not found: %EXECUTABLE_FILE%
    pause
    exit /b 1
)

if not exist "%SPEC_FILE%" (
    echo ERROR: Spec file not found: %SPEC_FILE%
    pause
    exit /b 1
)

echo [1/5] Building Music-Assassin v%VERSION%
echo.

REM Activate venv if available and not already activated
if not "!VENV_PATH!"=="" (
    if exist "!VENV_PATH!\Scripts\activate.bat" (
        call "!VENV_PATH!\Scripts\activate.bat"
    )
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

REM Check FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ERROR: FFmpeg not found
    echo Please install: https://ffmpeg.org/download.html
    pause
    exit /b 1
)

echo [2/5] Cleaning old builds for v%VERSION%
if exist "%BUILD_DIR%" (
    rd /s /q "%BUILD_DIR%" >nul 2>&1
)
echo Build directory cleaned.
echo.

echo [3/5] Building executable...
echo PyInstaller output will appear below:
echo ---

REM Build with version-specific output directory
pyinstaller "%SPEC_FILE%" ^
    --distpath "%BUILD_DIR%\dist" ^
    --workpath "%BUILD_DIR%\build" ^
    --specpath "%BUILD_DIR%"

if errorlevel 1 (
    echo ---
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo ---
echo.

echo [4/5] Verifying build...

if exist "%BUILD_DIR%\dist\MusicAssassin_v%VERSION%.exe" (
    echo.
    echo ============================================================
    echo   BUILD SUCCESSFUL!
    echo ============================================================
    echo.
    echo Version:    v%VERSION%
    echo Location:   %BUILD_DIR%\dist\MusicAssassin_v%VERSION%.exe
    
    powershell -NoProfile -Command "$f=Get-Item '%BUILD_DIR%\dist\MusicAssassin_v%VERSION%.exe'; Write-Host ('Size:       ' + [math]::round($f.length/1MB,1) + ' MB')"
    echo.
    
    REM Show version-specific info
    if "%VERSION%"=="0.0.1" (
        echo Version 0.0.1: Initial MoviePy version
        echo Uses: MoviePy import, Spleeter 2.1.0
    ) else if "%VERSION%"=="0.0.2" (
        echo Version 0.0.2: MoviePy with Logger
        echo Uses: MoviePy import with progress logging
    ) else if "%VERSION%"=="0.1.0" (
        echo Version 0.1.0: FFMPEG Transition
        echo Uses: FFMPEG for video/audio import, Spleeter 2.1.0
        echo Improvement: Much faster than MoviePy
    ) else if "%VERSION%"=="0.2.0" (
        echo Version 0.2.0: FFMPEG Cleanup ^& Optimization
        echo Uses: FFMPEG with spawn fix, Spleeter 2.2.0
        echo Improvement: Reduced memory usage, cleaned code
    ) else if "%VERSION%"=="0.3.3" (
        echo Version 0.3.3: Enhanced Edition
        echo Uses: Spleeter 4-stems, Harmonic Rejection, DSP filtering
        echo Supports: YouTube (experimental)
    ) else if "%VERSION%"=="0.4.1" (
        echo Version 0.4.1: DEMUCS/Spleeter Hybrid
        echo Uses: DEMUCS neural separation + Spleeter fallback
        echo Features: Advanced pre/post-filtering, 3 processing modes
    ) else if "%VERSION%"=="0.4.2" (
        echo Version 0.4.2: DEMUCS/Spleeter Hybrid (memory-optimised)
        echo Uses: Selective lazy imports, chunked streaming export
    ) else if "%VERSION%"=="0.5.0" (
        echo Version 0.5.0: CPU Streaming Edition (Current)
        echo Uses: DEMUCS only - htdemucs/htdemucs_ft/mdx_extra
        echo Features: Model cache, 60s chunks, INT8 quantisation, ETA
        echo Bundle: ~1.5 GB smaller than v0.4.x (no Spleeter/TF)
    )
    echo.
    
    echo To run:
    echo   %BUILD_DIR%\dist\MusicAssassin_v%VERSION%.exe
    echo.
    echo To distribute:
    echo   - Direct: Share the .exe file
    echo   - ZIP: Compress %BUILD_DIR%\dist\ folder
    echo.
    
    pause
) else (
    echo ERROR: Build verification failed
    echo Executable not found at: %BUILD_DIR%\dist\MusicAssassin_v%VERSION%.exe
    pause
    exit /b 1
)

echo [5/5] Build complete!
echo.
pause
