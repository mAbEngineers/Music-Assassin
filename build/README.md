# Music-Assassin Build System
## Build Scripts and PyInstaller Configurations

Quick links to version-specific build files and documentation.

---

## 📦 What's Here

This folder contains everything needed to build executables for any Music-Assassin version.

```
build/
├── build.sh                          (macOS/Linux universal build script)
├── build.bat                         (Windows universal build script)
├── MusicAssassin_v0.0.1.spec        (Config for v0.0.1)
├── MusicAssassin_v0.0.2.spec        (Config for v0.0.2)
├── MusicAssassin_v0.3.3.spec        (Config for v0.3.3)
├── MusicAssassin_v0.4.1.spec        (Config for v0.4.1)
├── README.md                         (This file)
└── output_v*/                        (Build outputs by version)
```

---

## 🚀 Quick Start

### Windows
```batch
build.bat 0.4.1
```

### macOS/Linux
```bash
chmod +x build.sh
./build.sh 0.4.1
```

### Result
Executable created in: `build/output_v0.4.1/dist/MusicAssassin_v0.4.1`

---

## 📋 Available Versions

| Version | Command | Type | Status |
|---------|---------|------|--------|
| **0.0.1** | `./build.sh 0.0.1` | MoviePy era | Legacy |
| **0.0.2** | `./build.sh 0.0.2` | MoviePy with logging | Legacy |
| **0.1.0** | `./build.sh 0.1.0` | FFMPEG transition | Legacy |
| **0.2.0** | `./build.sh 0.2.0` | FFMPEG cleanup | Legacy |
| **0.3.3** | `./build.sh 0.3.3` | Enhanced (Harmonic Rejection) | Stable |
| **0.4.1** | `./build.sh 0.4.1` | DEMUCS/Spleeter Hybrid | **Current** |

---

## 📖 Using Build Scripts

### Full syntax

**macOS/Linux:**
```bash
./build.sh <version>
```

**Windows:**
```batch
build.bat <version>
```

### Examples

Build different versions:
```bash
./build.sh 0.0.1   # MoviePy initial version
./build.sh 0.0.2   # MoviePy with logger
./build.sh 0.1.0   # FFMPEG transition
./build.sh 0.2.0   # FFMPEG cleanup & optimization
./build.sh 0.3.3   # Enhanced with YouTube
./build.sh 0.4.1   # Latest DEMUCS/Spleeter
```

### What Happens

1. ✅ Validates Python & FFmpeg
2. ✅ Cleans old builds
3. ✅ Runs PyInstaller with version-specific .spec
4. ✅ Creates versioned output directory
5. ✅ Verifies executable created
6. ✅ Shows path and size

### Output

Each version gets its own directory:
```
build/output_v0.4.1/dist/MusicAssassin_v0.4.1         (macOS/Linux)
build/output_v0.4.1/dist/MusicAssassin_v0.4.1.exe     (Windows)
```

---

## 🔧 .Spec Files Explained

Each version has a PyInstaller spec file with specific configuration:

### MusicAssassin_v0.0.1.spec
- **Uses:** MoviePy, Spleeter 2.1.0
- **Size:** ~500MB
- **Legacy:** Early development version

### MusicAssassin_v0.0.2.spec
- **Uses:** MoviePy, Spleeter 2.1.0, Logger
- **Size:** ~500MB
- **Legacy:** Added progress logging

### MusicAssassin_v0.3.3.spec
- **Uses:** Spleeter 4-stems, DSP filtering, YouTube support
- **Includes:** Harmonic rejection, notch filters, formant enhancement
- **Size:** ~1.5GB
- **Status:** Stable, production-ready

### MusicAssassin_v0.4.1.spec (Latest)
- **Uses:** DEMUCS (recommended) + Spleeter (fallback)
- **Features:** Advanced pre/post-filtering, 3 processing modes
- **Includes:** PyTorch, DEMUCS models, Spleeter, TensorFlow
- **Size:** ~2-2.5GB
- **Status:** Current, recommended

---

## ⚙️ Prerequisites for All Versions

Required before building:

- ✅ **Python 3.6+** (macOS/Linux) or **3.8+** (all versions recommended)
- ✅ **FFmpeg** (system package, not pip)
- ✅ **5GB free disk space**
- ✅ **Dependencies installed** (see requirements.txt)

### Install Prerequisites

**macOS:**
```bash
brew install python ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install python3 ffmpeg
```

**Windows:**
- Python: https://www.python.org
- FFmpeg: https://ffmpeg.org/download.html
  - Or: `choco install ffmpeg`

---

## 💾 Build Outputs

After building, each version has:

```
build/output_v0.4.1/
├── dist/
│   ├── MusicAssassin_v0.4.1          (executable - macOS/Linux)
│   ├── MusicAssassin_v0.4.1.exe      (executable - Windows)
│   └── _internal/                    (support files - KEEP THIS!)
│       ├── demucs/
│       ├── spleeter/
│       ├── torch/
│       └── ...
├── build/                            (temporary PyInstaller files)
└── MusicAssassin_v0.4.1.spec        (copy of spec used)
```

**Important:** Don't delete `_internal/` folder - it's required!

---

## 🎯 Version Comparison

| Feature | v0.0.1 | v0.0.2 | v0.1.0 | v0.2.0 | v0.3.3 | v0.4.1 |
|---------|--------|--------|--------|--------|--------|--------|
| **Engine** | MoviePy | MoviePy | FFMPEG | FFMPEG | Spleeter | DEMUCS/Spleeter |
| **Quality** | Basic | Basic | Basic | Basic | Good | Excellent |
| **Size** | ~500MB | ~500MB | ~500MB | ~500MB | ~1.5GB | ~2GB |
| **Speed** | Slow | Slow | Medium | Medium | Medium | Medium-Fast |
| **YouTube** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **DSP Filters** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Modes** | 1 | 1 | 1 | 1 | 3 | 3 |
| **Status** | Legacy | Legacy | Legacy | Legacy | Stable | **Current** |

---

## 🛠️ Manual Build (Advanced)

If you want to manually run PyInstaller:

```bash
# macOS/Linux
pyinstaller build/MusicAssassin_v0.4.1.spec

# Windows
pyinstaller build\MusicAssassin_v0.4.1.spec
```

Or with custom output:

```bash
pyinstaller build/MusicAssassin_v0.4.1.spec \
    --distpath build/output_v0.4.1/dist \
    --buildpath build/output_v0.4.1/build
```

---

## 📊 Build Times

Typical build times (first run):

| Step | Time |
|------|------|
| Install dependencies | 5-10 min |
| PyInstaller analysis | 5 min |
| Bundle creation | 10-20 min |
| Finalization | 3-5 min |
| **Total** | **25-50 min** |

Subsequent builds (cached): 15-25 minutes

---

## 🐛 Troubleshooting

### Build script won't run

**macOS/Linux:**
```bash
chmod +x build.sh
./build.sh 0.4.1
```

**Windows:**
- Run Command Prompt or PowerShell **as Administrator**
- Then: `build.bat 0.4.1`

### "FFmpeg not found"

Install FFmpeg:
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html

### "Python not found"

Install Python 3.8+:
- **All:** https://www.python.org
- **Windows:** Ensure "Add Python to PATH" is checked during install

### Build hangs / freezes

**Normal:** PyInstaller with PyTorch/TensorFlow takes 20-50 minutes

If genuinely stuck (>2 hours):
```bash
# Cancel build
Ctrl+C

# Try again with clean slate
rm -rf build/output_v0.4.1
./build.sh 0.4.1
```

### Out of disk space

Need 5GB+ free. Check:
```bash
df -h /          # macOS/Linux
wmic logicaldisk get freespace  # Windows
```

### For more help
See: [BUILD_TROUBLESHOOTING.md](../BUILD_TROUBLESHOOTING.md) (parent directory)

---

## 📤 Distribution

After successful build, distribute the executable:

### Option 1: Standalone File
Just share: `build/output_v0.4.1/dist/MusicAssassin_v0.4.1.exe`

### Option 2: ZIP Archive
```bash
# macOS/Linux
tar -czf Music-Assassin-v0.4.1.tar.gz build/output_v0.4.1/dist/

# Windows
Compress-Archive -Path build\output_v0.4.1\dist -DestinationPath Music-Assassin-v0.4.1.zip
```

### Option 3: Installer
Use Inno Setup or similar (see parent BUILD_TROUBLESHOOTING.md)

---

## 🔄 Building Multiple Versions

Build several versions:

```bash
# macOS/Linux
./build.sh 0.0.1
./build.sh 0.3.3
./build.sh 0.4.1

# Windows
build.bat 0.0.1
build.bat 0.3.3
build.bat 0.4.1
```

Each goes to its own directory:
```
build/output_v0.0.1/
build/output_v0.3.3/
build/output_v0.4.1/
```

---

## 📝 Version Info

| Version | Release | Type | Features |
|---------|---------|------|----------|
| 0.0.1 | 2020 | MoviePy | Initial version |
| 0.0.2 | 2020 | MoviePy | Logging support |
| 0.3.3 | 2023 | Spleeter | Harmonic rejection, UI redesign, YouTube (exp.) |
| 0.4.1 | 2024 | Hybrid | DEMUCS/Spleeter, advanced filtering (current) |

---

## 🎓 Learning Path

### Want to understand the build process?

1. Run: `./build.sh 0.4.1`
2. Watch output as PyInstaller bundles files
3. Check `build/output_v0.4.1/dist/_internal/` to see bundled files
4. Look at `MusicAssassin_v0.4.1.spec` to see configuration

### Want to customize a build?

1. Copy a .spec file: `cp MusicAssassin_v0.4.1.spec my_custom.spec`
2. Edit with your changes
3. Run PyInstaller: `pyinstaller my_custom.spec`

### Want to add an icon?

Edit the .spec file:
```python
exe = EXE(
    ...
    icon='path/to/icon.ico',  # Add your icon
    ...
)
```

Then rebuild.

---

## 📚 Documentation

Related files (parent directory):
- [BUILD_QUICK_START.md](../BUILD_QUICK_START.md) - Quick start guide
- [BUILD_TROUBLESHOOTING.md](../BUILD_TROUBLESHOOTING.md) - Problem solutions
- [PACKAGING_MANUAL.md](../PACKAGING_MANUAL.md) - Detailed guide
- [BUILD_INDEX.md](../BUILD_INDEX.md) - Documentation index

---

## 💡 Tips & Tricks

### Speed up subsequent builds
```bash
pip install -U --no-deps pyinstaller
```

### Check what's in _internal
```bash
ls -lR build/output_v0.4.1/dist/_internal/ | head -50
```

### Verify executable works before distributing
```bash
./build/output_v0.4.1/dist/MusicAssassin_v0.4.1

# Or on Windows
build\output_v0.4.1\dist\MusicAssassin_v0.4.1.exe
```

### Get file size
```bash
du -h build/output_v0.4.1/dist/MusicAssassin_v0.4.1
# Or (Windows)
powershell "Get-Item build\output_v0.4.1\dist\MusicAssassin_v0.4.1.exe | Select-Object -ExpandProperty Length"
```

---

## ❓ FAQ

**Q: Why is the executable so large (~2GB)?**  
A: PyTorch (~600MB) + TensorFlow (~500MB) + DEMUCS models (~300MB) + NumPy/SciPy + dependencies

**Q: Can I reduce the size?**  
A: Build v0.0.1 or v0.0.2 for smaller executables (~500MB), or v0.3.3 (~1.5GB)

**Q: Do users need Python installed?**  
A: No, the executable includes Python. Just need FFmpeg (system package).

**Q: Can I build for a different OS?**  
A: PyInstaller builds for the current OS. Run on Windows to build .exe, on macOS to build .app, etc.

**Q: How do I add my custom code?**  
A: Modify `MusicAssassinGUI_v0.4.1.py`, then rebuild.

---

## 🚀 Ready?

```bash
# macOS/Linux
chmod +x build.sh
./build.sh 0.4.1

# Windows
build.bat 0.4.1
```

Your executable will be ready in ~30-50 minutes!

---

*Music-Assassin v0.4.1 - Build System*
