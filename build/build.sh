#!/bin/bash

# Music-Assassin Build Script - Version Selector
# Builds any version specified by the user

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Virtual Environment Path
# Can be set as environment variable: VENV_PATH=/path/to/venv ./build.sh <version>
# Or will auto-detect from common locations
if [ -z "$VENV_PATH" ]; then
    # Try common locations
    if [ -d "$HOME/Documents/venvs/call_venv" ]; then
        VENV_PATH="$HOME/Documents/venvs/call_venv"
    elif [ -d "$HOME/Documents/venvs/assassin_venv" ]; then
        VENV_PATH="$HOME/Documents/venvs/assassin_venv"
    elif [ -d "../venvs/call_venv" ]; then
        VENV_PATH="../venvs/call_venv"
    elif [ -d "../venvs/assassin_venv" ]; then
        VENV_PATH="../venvs/assassin_venv"
    elif [ -d "venv" ]; then
        VENV_PATH="venv"
    elif [ -d ".venv" ]; then
        VENV_PATH=".venv"
    fi
fi

echo ""
echo "============================================================"
echo -e "${BLUE}  Music Assassin - PyInstaller Build Script${NC}"
echo "============================================================"
echo ""

# Check version argument
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage: ./build.sh <version>${NC}"
    echo ""
    echo "Available versions:"
    echo "  ./build.sh 0.0.1   - Initial MoviePy version"
    echo "  ./build.sh 0.0.2   - MoviePy with Logger"
    echo "  ./build.sh 0.1.0   - FFMPEG Transition"
    echo "  ./build.sh 0.2.0   - FFMPEG Cleanup & Optimization"
    echo "  ./build.sh 0.3.3   - Enhanced Edition (Harmonic Rejection)"
    echo "  ./build.sh 0.3.4   - YouTube Edition"
    echo "  ./build.sh 0.4.0   - DEMUCS only (chunked, early prototype)"
    echo "  ./build.sh 0.4.1   - DEMUCS + Spleeter (chunked, INT8, full DSP)"
    echo "  ./build.sh 0.4.2   - DEMUCS + Spleeter + MDX-Net (vocals-only MDX-Net)"
    echo "  ./build.sh 0.4.4   - + VAD gate + DeepFilterNet cleanup (Recommended)"
    echo "  ./build.sh 0.5.0   - CPU Streaming Edition"
    echo ""
    echo "Examples:"
    echo "  ./build.sh 0.5.0"
    echo "  VENV_PATH=/path/to/venv ./build.sh 0.4.1"
    echo ""
    exit 1
fi

# Strip leading 'v' if provided (e.g. v0.4.1 -> 0.4.1)
VERSION="${1#v}"

EXECUTABLE_FILE="versions/MusicAssassinGUI_v${VERSION}.py"
SPEC_FILE="build/MusicAssassin_v${VERSION}.spec"
BUILD_DIR="build/output_v${VERSION}"

# Verify input is valid
case $VERSION in
    0.0.1|0.0.2|0.1.0|0.2.0|0.3.3|0.3.4|0.4.0|0.4.1|0.4.2|0.4.4|0.5.0)
        # Valid version
        ;;
    *)
        echo -e "${RED}ERROR: Invalid version '${VERSION}'${NC}"
        echo "Valid versions: 0.0.1, 0.0.2, 0.1.0, 0.2.0, 0.3.3, 0.3.4, 0.4.0, 0.4.1, 0.4.2, 0.4.4, 0.5.0"
        exit 1
        ;;
esac

# Check if files exist
if [ ! -f "$EXECUTABLE_FILE" ]; then
    echo -e "${RED}ERROR: Python file not found: $EXECUTABLE_FILE${NC}"
    exit 1
fi

if [ ! -f "$SPEC_FILE" ]; then
    echo -e "${RED}ERROR: Spec file not found: $SPEC_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}[1/6]${NC} Building Music-Assassin v${VERSION}"
echo ""

# Try to detect and activate venv
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
    fi
fi

# Use venv python if available, otherwise fall back to python3
PYTHON=$(command -v python 2>/dev/null || command -v python3 2>/dev/null)
if [ -z "$PYTHON" ]; then
    echo -e "${RED}ERROR: Python not found${NC}"
    exit 1
fi
echo -e "${GREEN}Python:${NC} $($PYTHON --version 2>&1)"

# Check if PyInstaller is available
if ! "$PYTHON" -m PyInstaller --version &> /dev/null; then
    echo -e "${RED}ERROR: PyInstaller not installed in current Python environment${NC}"
    echo "Install with: pip install pyinstaller"
    exit 1
fi

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}ERROR: FFmpeg not found${NC}"
    echo "Install with:"
    echo "  macOS:  brew install ffmpeg"
    echo "  Linux:  sudo apt-get install ffmpeg"
    exit 1
fi

echo -e "${GREEN}[2/6]${NC} Cleaning old builds for v${VERSION}..."
rm -rf "$BUILD_DIR" build/__pycache__ build/*.pyc 2>/dev/null || true
echo "Build directory cleaned."
echo ""

echo -e "${GREEN}[3/6]${NC} Building executable..."
echo "PyInstaller output will appear below:"
echo "---"

# Build with version-specific output directory
"$PYTHON" -m PyInstaller "$SPEC_FILE" \
    --distpath "$BUILD_DIR/dist" \
    --workpath "$BUILD_DIR/build" \
    2>&1

if [ $? -ne 0 ]; then
    echo "---"
    echo -e "${RED}ERROR: Build failed${NC}"
    exit 1
fi

echo "---"
echo ""

# Verify build
echo -e "${GREEN}[4/6]${NC} Verifying build..."

executable_path="$BUILD_DIR/dist/MusicAssassin_v${VERSION}"

if [ -f "$executable_path" ]; then
    size=$(du -h "$executable_path" | cut -f1)
    
    echo ""
    echo "============================================================"
    echo -e "${GREEN}  BUILD SUCCESSFUL!${NC}"
    echo "============================================================"
    echo ""
    echo -e "${BLUE}Version:${NC}    v${VERSION}"
    echo -e "${BLUE}Location:${NC}   $executable_path"
    echo -e "${BLUE}Size:${NC}       $size"
    echo ""
    
    chmod +x "$executable_path"
    echo -e "${GREEN}✓${NC} Executable permissions set"
    echo ""
    
    # Show version-specific info
    case $VERSION in
        0.0.1)
            echo "Version 0.0.1: Initial MoviePy version"
            echo "Uses: MoviePy import, Spleeter 2.1.0"
            echo ""
            ;;
        0.0.2)
            echo "Version 0.0.2: MoviePy with Logger"
            echo "Uses: MoviePy import with progress logging"
            echo ""
            ;;
        0.1.0)
            echo "Version 0.1.0: FFMPEG Transition"
            echo "Uses: FFMPEG for video/audio import, Spleeter 2.1.0"
            echo "Improvement: Much faster than MoviePy"
            echo ""
            ;;
        0.2.0)
            echo "Version 0.2.0: FFMPEG Cleanup & Optimization"
            echo "Uses: FFMPEG with spawn fix, Spleeter 2.2.0"
            echo "Improvement: Reduced memory usage, cleaned code"
            echo ""
            ;;
        0.3.3)
            echo "Version 0.3.3: Enhanced Edition"
            echo "Uses: Spleeter 4-stems, Harmonic Rejection, DSP filtering"
            echo "Supports: YouTube (experimental)"
            echo ""
            ;;
        0.4.0)
            echo "Version 0.4.0: DEMUCS only (early prototype)"
            echo "Uses: DEMUCS chunked async processing, torchaudio pipeline"
            echo ""
            ;;
        0.4.1)
            echo "Version 0.4.1: DEMUCS + Spleeter (chunked)"
            echo "Uses: DEMUCS + Spleeter 4-stems, INT8 quantisation, model cache"
            echo "Fix: Pre-chunked waveform (60s) — htdemucs_ft/mdx_extra no longer OOM"
            echo ""
            ;;
        0.4.2)
            echo "Version 0.4.2: DEMUCS + Spleeter + MDX-Net (Current)"
            echo "Uses: MDX-Net (UVR) vocals-only models via audio-separator (onnxruntime)"
            echo "Features: UVR-MDX-NET-Voc_FT + Kim_Vocal_2, fast on CPU, auto-download"
            echo "Requires: Python 3.9+ venv (audio-separator constraint)"
            echo "Install deps: pip install -r requirements/requirements_mdxnet_only.txt"
            echo ""
            ;;
        0.4.4)
            echo "Version 0.4.4: DEMUCS + MDX-Net + VAD + DeepFilterNet (Recommended)"
            echo "Uses: Demucs/MDX-Net separation + Silero VAD silence gate + optional DeepFilterNet cleanup"
            echo "Fix: global (not per-chunk) normalization, last-chunk fade-to-zero padding"
            echo "Note: Spleeter/TensorFlow excluded from this build (PyInstaller+keras crash);"
            echo "      run from source with setup_v044.sh if you need the Spleeter engine."
            echo "Install deps: bash setup_v044.sh [venv_path]"
            echo ""
            ;;
        0.5.0)
            echo "Version 0.5.0: CPU Streaming Edition (Current)"
            echo "Uses: DEMUCS only (no TensorFlow) — htdemucs/htdemucs_ft/mdx_extra"
            echo "Features: Model cache, 60 s outer chunks, INT8 quantisation, ETA"
            echo "Bundle savings: ~1.5 GB smaller (no Spleeter/TF)"
            echo ""
            ;;
    esac
    
    echo "To run:"
    echo "  $executable_path"
    echo ""
    echo "To distribute:"
    echo "  - Direct: Share the executable"
    echo "  - ZIP: tar -czf Music-Assassin-v${VERSION}.tar.gz $BUILD_DIR/dist/"
    echo ""

else
    echo -e "${RED}ERROR: Build verification failed${NC}"
    echo "Executable not found at: $executable_path"
    exit 1
fi

# ----------------------------------------------------------------
# .deb packaging (Ubuntu/Debian only)
# ----------------------------------------------------------------
echo -e "${GREEN}[5/5]${NC} Packaging as .deb (Ubuntu/Debian)..."

if command -v dpkg-deb &> /dev/null; then
    DEB_ROOT="$BUILD_DIR/deb_pkg"
    DEB_BIN_DIR="$DEB_ROOT/usr/bin"
    DEB_CTRL_DIR="$DEB_ROOT/DEBIAN"

    rm -rf "$DEB_ROOT"
    mkdir -p "$DEB_BIN_DIR" "$DEB_CTRL_DIR"

    cp "$executable_path" "$DEB_BIN_DIR/music-assassin"
    chmod 755 "$DEB_BIN_DIR/music-assassin"

    cat > "$DEB_CTRL_DIR/control" <<EOF
Package: music-assassin
Version: ${VERSION}
Architecture: amd64
Maintainer: Music-Assassin <noreply@music-assassin.local>
Description: Music-Assassin v${VERSION}
 AI-powered vocal and instrument separator (DEMUCS/Spleeter).
EOF

    DEB_OUT="$BUILD_DIR/dist/MusicAssassin_v${VERSION}.deb"
    # Use -Znone: the binary is already UPX-compressed; recompressing a 3+ GB
    # file with zstd (dpkg 1.21+ default) silently produces a broken archive.
    fakeroot dpkg-deb --build -Znone "$DEB_ROOT" "$DEB_OUT"

    if [ $? -eq 0 ]; then
        deb_size=$(du -h "$DEB_OUT" | cut -f1)
        echo -e "${GREEN}✓${NC} .deb package: $DEB_OUT ($deb_size)"
        echo ""
        echo "Install with:"
        echo "  sudo dpkg -i $DEB_OUT"
        echo ""
    else
        echo -e "${RED}WARNING: .deb packaging failed (executable still available)${NC}"
        echo ""
    fi
else
    echo -e "${YELLOW}WARNING: dpkg-deb not found — skipping .deb packaging${NC}"
    echo "Install with: sudo apt-get install dpkg"
    echo ""
fi

echo -e "${GREEN}[6/6]${NC} Build complete!"
echo ""
