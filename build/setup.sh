#!/bin/bash

# Music-Assassin Build System Setup
# Makes build scripts executable and checks prerequisites

echo ""
echo "============================================================"
echo "  Music-Assassin Build System Setup"
echo "============================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Make scripts executable
echo "Making build scripts executable..."
chmod +x build/build.sh 2>/dev/null
chmod +x build/*.sh 2>/dev/null
echo -e "${GREEN}✓${NC} Build scripts ready"
echo ""

# Check Python
echo "Checking prerequisites..."
if command -v python3 &> /dev/null; then
    python_ver=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Python $python_ver found"
else
    echo -e "${RED}✗${NC} Python not found (install Python 3.8+)"
fi

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✓${NC} FFmpeg found"
else
    echo -e "${RED}✗${NC} FFmpeg not found (install from https://ffmpeg.org)"
fi

# Check PyInstaller
if pip3 list 2>/dev/null | grep -q pyinstaller; then
    echo -e "${GREEN}✓${NC} PyInstaller installed"
else
    echo -e "${YELLOW}!${NC} PyInstaller not installed"
    echo "   Run: pip install -r requirements_packaging.txt"
fi

echo ""
echo "============================================================"
echo "  Build System Ready!"
echo "============================================================"
echo ""
echo "Available versions:"
echo "  ./build/build.sh 0.0.1   - Initial MoviePy version"
echo "  ./build/build.sh 0.0.2   - MoviePy with Logger"
echo "  ./build/build.sh 0.1.0   - FFMPEG Transition"
echo "  ./build/build.sh 0.2.0   - FFMPEG Cleanup & Optimization"
echo "  ./build/build.sh 0.3.3   - Enhanced Edition (Harmonic Rejection)"
echo "  ./build/build.sh 0.4.1   - DEMUCS/Spleeter Hybrid (Current)"
echo ""
echo "Example:"
echo "  ./build/build.sh 0.4.1"
echo ""
echo "For more information:"
echo "  cat build/README.md"
echo ""
