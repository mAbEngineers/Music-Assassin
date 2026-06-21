# Build System - Complete File Index

Master reference for all build-related files and documentation.

## 📁 File Organization

### Root Directory
Build-related documentation files in main directory:

| File | Purpose | Read When |
|------|---------|-----------|
| **BUILD_INDEX.md** | Navigation & overview | First time |
| **BUILD_QUICK_START.md** | 3-step quick guide | Want fast results |
| **BUILD_TROUBLESHOOTING.md** | Problem solutions | Errors occur |
| **PACKAGING_MANUAL.md** | 500+ line detailed guide | Need detailed info |
| **BUILD_FOLDER_STRUCTURE.md** | Folder overview & quick ref | Understanding structure |
| **BUILD_MIGRATION_SUMMARY.md** | What was created & why | Understand changes |
| **REQUIREMENTS_GUIDE.md** | Dependency documentation | Understanding deps |
| **requirements.txt** | Main dependencies | Installation |
| **requirements_packaging.txt** | Build tools | `pip install -r requirements_packaging.txt` |
| **requirements_latest.txt** | Latest of all packages | Alternative |
| **requirements_pinned.txt** | Exact versions | Production |
| **requirements_demucs_only.txt** | DEMUCS only variant | Lightweight |
| **requirements_spleeter_only.txt** | Spleeter only variant | Lightweight |
| **requirements_by_version.txt** | Historical versions | Reference |

### build/ Directory
Build scripts and configurations:

| File | Purpose | Platform |
|------|---------|----------|
| **build/README.md** | Build folder master guide | All |
| **build/setup.sh** | Setup & verification | macOS/Linux |
| **build/setup.bat** | Setup & verification | Windows |
| **build/build.sh** | Universal build script | macOS/Linux |
| **build/build.bat** | Universal build script | Windows |
| **build/MusicAssassin_v0.0.1.spec** | PyInstaller config | All (v0.0.1) |
| **build/MusicAssassin_v0.0.2.spec** | PyInstaller config | All (v0.0.2) |
| **build/MusicAssassin_v0.3.3.spec** | PyInstaller config | All (v0.3.3) |
| **build/MusicAssassin_v0.4.1.spec** | PyInstaller config | All (v0.4.1) |
| **build/output_v0.0.1/** | Build output | Auto-created |
| **build/output_v0.0.2/** | Build output | Auto-created |
| **build/output_v0.3.3/** | Build output | Auto-created |
| **build/output_v0.4.1/** | Build output | Auto-created |

## 🚀 Getting Started

### Step 1: Read Overview
Choose based on your need:

- **Beginner:** `BUILD_QUICK_START.md` - 5 minute read
- **Detailed:** `build/README.md` - 10 minute read
- **Deep dive:** `PACKAGING_MANUAL.md` - 30 minute read

### Step 2: Setup Environment

**macOS/Linux:**
```bash
chmod +x build/setup.sh
./build/setup.sh
```

**Windows:**
```batch
build\setup.bat
```

### Step 3: Build Executable

**Build latest version (v0.4.1):**
```bash
./build/build.sh 0.4.1        # macOS/Linux
build\build.bat 0.4.1         # Windows
```

**Build other version:**
```bash
./build/build.sh 0.0.1    # or 0.0.2, 0.3.3, 0.4.1
build\build.bat 0.0.1     # Windows
```

## 📖 Documentation Reading Guide

### For Quick Build (10 minutes)
1. `BUILD_QUICK_START.md` (read once)
2. Run `./build/build.sh 0.4.1`
3. Done!

### For Detailed Understanding (20 minutes)
1. `BUILD_FOLDER_STRUCTURE.md` (overview)
2. `build/README.md` (build folder guide)
3. Try building a version

### For Complete Knowledge (1 hour)
1. `BUILD_INDEX.md` (navigation)
2. `PACKAGING_MANUAL.md` (detailed guide)
3. `BUILD_TROUBLESHOOTING.md` (problems)
4. Build multiple versions

### For Distribution (30 minutes)
1. `BUILD_QUICK_START.md` (build)
2. "Distribution" section in `PACKAGING_MANUAL.md`
3. Create installer or ZIP

### For Troubleshooting (as needed)
1. `BUILD_TROUBLESHOOTING.md` (find your issue)
2. Follow solution
3. Retry

## 📚 Documentation Map

```
START HERE
    ↓
[BUILD_QUICK_START.md] ← For fast results
    ↓
[build/README.md] ← Master build guide
    ├─→ [BUILD_FOLDER_STRUCTURE.md] ← Folder layout
    ├─→ [build/build.sh / build.bat] ← Actual scripts
    ├─→ [build/*.spec] ← Config files
    └─→ [build/output_*] ← Build outputs
    ↓
[PACKAGING_MANUAL.md] ← Detailed guide
    ├─→ Advanced options
    ├─→ Distribution methods
    └─→ Installer creation
    ↓
[BUILD_TROUBLESHOOTING.md] ← When things go wrong
    ├─→ Error solutions
    ├─→ Platform-specific
    └─→ Diagnostic commands
    ↓
[BUILD_INDEX.md] ← Navigation & reference
    ↓
[BUILD_MIGRATION_SUMMARY.md] ← What was created
```

## 🔍 Finding What You Need

### "I want to build an executable"
→ `BUILD_QUICK_START.md` (easiest)  
→ or `build/README.md` (more control)

### "I have an error during build"
→ `BUILD_TROUBLESHOOTING.md` (search your error)

### "I want to understand the build system"
→ `BUILD_FOLDER_STRUCTURE.md` (overview)  
→ `PACKAGING_MANUAL.md` (details)

### "I want to build multiple versions"
→ `build/README.md` → "Building Multiple Versions"

### "I want to distribute the executable"
→ `PACKAGING_MANUAL.md` → "Distribution" section

### "I want to customize the build"
→ `PACKAGING_MANUAL.md` → "Advanced Options"

### "I need to understand the structure"
→ `BUILD_MIGRATION_SUMMARY.md` (what was created)  
→ `BUILD_FOLDER_STRUCTURE.md` (folder layout)

## 📋 Build Commands Reference

### Setup (First Time)
```bash
# macOS/Linux
chmod +x build/setup.sh
./build/setup.sh

# Windows
build\setup.bat
```

### Build Version
```bash
# macOS/Linux
./build/build.sh 0.4.1       # v0.4.1
./build/build.sh 0.3.3       # v0.3.3
./build/build.sh 0.0.1       # v0.0.1

# Windows
build\build.bat 0.4.1
build\build.bat 0.3.3
build\build.bat 0.0.1
```

### Manual Build (Advanced)
```bash
# macOS/Linux
pyinstaller build/MusicAssassin_v0.4.1.spec

# Windows
pyinstaller build\MusicAssassin_v0.4.1.spec
```

### Find Executable
```bash
# macOS/Linux
ls -lh build/output_v0.4.1/dist/MusicAssassin_v0.4.1

# Windows
dir build\output_v0.4.1\dist\MusicAssassin_v0.4.1.exe
```

## 🎯 Common Tasks

### Build Latest Version
`BUILD_QUICK_START.md` → Run `./build/build.sh 0.4.1`

### Build Earlier Version
`build/README.md` → Version Comparison table → Run `./build/build.sh 0.3.3`

### Fix Build Error
`BUILD_TROUBLESHOOTING.md` → Find error → Follow solution

### Create Windows Installer
`PACKAGING_MANUAL.md` → "Distribution" → "Windows: Using Inno Setup"

### Build Smaller Executable
`PACKAGING_MANUAL.md` → "Advanced Options" → "Optimize for Smaller Size"

### Add Custom Icon
`PACKAGING_MANUAL.md` → "Advanced Options" → "Add Custom Icon"

### View File Sizes
`BUILD_FOLDER_STRUCTURE.md` → "Version Sizes" table

## 📊 Quick Reference

### Versions Available
```
v0.0.1 - Initial MoviePy (legacy, ~500MB)
v0.0.2 - MoviePy Logger (legacy, ~500MB)
v0.3.3 - Enhanced Edition (stable, ~1.5GB)
v0.4.1 - DEMUCS/Spleeter (current, ~2GB)
```

### Check Installation
```bash
python --version        # Need 3.8+
ffmpeg -version        # Must be installed
pip list | grep -i pyinstaller  # Check PyInstaller
```

### Common Issues & Solution Docs
```
FFmpeg not found        → BUILD_TROUBLESHOOTING.md #1
Python not found        → BUILD_TROUBLESHOOTING.md #2
Build hangs            → BUILD_TROUBLESHOOTING.md #3
No disk space          → BUILD_TROUBLESHOOTING.md #4
Module not found       → BUILD_TROUBLESHOOTING.md #5+
```

## 🔗 Cross-References

### In PACKAGING_MANUAL.md:

- Prerequisites
- Build Methods
- Output Files
- Troubleshooting
- Advanced Options
- Distribution
- Release Notes

### In BUILD_TROUBLESHOOTING.md:

- FFmpeg Issues (#1)
- Python/Environment (#2)
- PyInstaller Issues (#3)
- Disk Space (#4)
- Build Output (#5)
- GPU/CUDA (#6)
- Windows-Specific (#7)
- macOS-Specific (#8)
- Linux-Specific (#9)
- Network Issues (#10)
- Diagnostic Commands (#11)
- Full Clean Rebuild (#12)

### In PACKAGING_MANUAL.md:

- Prerequisites
- Quick Start
- Build Commands
- Output
- Troubleshooting
- Advanced Options
- Build Script
- Verification

## 🎓 Learning Path

**For Beginners:**
1. `BUILD_QUICK_START.md`
2. `./build/build.sh 0.4.1`
3. Test executable
4. Done!

**For Intermediate:**
1. `BUILD_FOLDER_STRUCTURE.md`
2. `build/README.md`
3. Build multiple versions
4. Try distribution options

**For Advanced:**
1. `PACKAGING_MANUAL.md` (all sections)
2. `BUILD_TROUBLESHOOTING.md` (all sections)
3. Modify .spec files
4. Create custom installers

## ✅ File Checklist

### Required Files
- ✅ `build/build.sh` - Build script
- ✅ `build/build.bat` - Build script
- ✅ `build/*.spec` - Config files (all 4 versions)
- ✅ `requirements.txt` - Dependencies

### Helpful Documentation
- ✅ `BUILD_QUICK_START.md` - Fast guide
- ✅ `build/README.md` - Build folder guide
- ✅ `PACKAGING_MANUAL.md` - Complete guide
- ✅ `BUILD_TROUBLESHOOTING.md` - Solutions
- ✅ `BUILD_FOLDER_STRUCTURE.md` - Overview

### Output Files (Auto-Created on Build)
- `build/output_v0.0.1/dist/` - Build output
- `build/output_v0.0.2/dist/` - Build output
- `build/output_v0.3.3/dist/` - Build output
- `build/output_v0.4.1/dist/` - Build output

## 🚀 Next Steps

1. **Choose version to build:**
   - Latest: v0.4.1 (DEMUCS/Spleeter)
   - Stable: v0.3.3 (Harmonic Rejection)
   - Lightweight: v0.0.1 or v0.0.2

2. **Run setup (if first time):**
   ```bash
   ./build/setup.sh          # macOS/Linux
   build\setup.bat           # Windows
   ```

3. **Build executable:**
   ```bash
   ./build/build.sh 0.4.1    # macOS/Linux
   build\build.bat 0.4.1     # Windows
   ```

4. **Find your executable:**
   - `build/output_v0.4.1/dist/MusicAssassin_v0.4.1` (Unix)
   - `build/output_v0.4.1/dist/MusicAssassin_v0.4.1.exe` (Windows)

5. **Use or distribute:**
   - Run directly
   - Create ZIP archive
   - Build Windows installer
   - Share with users

---

**Everything you need to build Music-Assassin executables is here!**

For quick start: `BUILD_QUICK_START.md` → `./build/build.sh 0.4.1`

*Music-Assassin Build System - Complete Reference*
