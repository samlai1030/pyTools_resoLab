# pyTools_ResoLab - EXE Packaging Guide

## Overview

This guide explains how to package the pyTools_ResoLab application as a standalone Windows .exe executable that can be distributed without requiring Python installation.

## Prerequisites

You need to have Python 3.8+ installed on your development machine.

## Installation Steps

### 1. Install PyInstaller

```bash
pip install pyinstaller
```

### 2. Navigate to Project Directory

```bash
cd /Users/samlai/Local_2/pyTools_resoLab
```

### 3. Build the Executable

#### Option A: Using the Build Script (Recommended)
```bash
# Make the script executable (first time only)
chmod +x build_exe.sh

# Run the build script
./build_exe.sh
```

#### Option B: Using PyInstaller Directly
```bash
pyinstaller --onefile \
    --windowed \
    --name pyTools_ResoLab \
    --add-data "makedown:makedown" \
    --hidden-import=PyQt5.sip \
    main.py
```

#### Option C: Using the Spec File
```bash
pyinstaller pyTools_ResoLab.spec
```

## Build Output

After building, you'll get:

```
dist/
├── pyTools_ResoLab.exe          # Single executable file (--onefile mode)
└── pyTools_ResoLab/             # Full distribution folder
    ├── pyTools_ResoLab.exe
    ├── ...dependencies...
    └── ...libraries...
```

## Output Files

### Single File Mode (--onefile)
- **File**: `dist/pyTools_ResoLab.exe`
- **Size**: ~200-300 MB
- **Advantage**: Single file, easy to distribute
- **Disadvantage**: Slower startup (unpacks at runtime)

### Folder Mode (default)
- **Folder**: `dist/pyTools_ResoLab/`
- **Contains**: .exe + all dependencies
- **Advantage**: Faster startup
- **Disadvantage**: Multiple files to distribute

## Usage

### On Your Development Machine
```bash
# Run directly from dist folder
dist/pyTools_ResoLab/pyTools_ResoLab.exe

# Or single file
dist/pyTools_ResoLab.exe
```

### On Another Windows Machine
1. Copy the entire `dist/pyTools_ResoLab/` folder OR `dist/pyTools_ResoLab.exe`
2. No Python installation required
3. Run `pyTools_ResoLab.exe`

## Customization

### Add App Icon
1. Create a 256x256 icon file (icon.ico)
2. Modify the PyInstaller command:
```bash
pyinstaller --onefile \
    --windowed \
    --icon=icon.ico \
    --name pyTools_ResoLab \
    main.py
```

### Include Additional Files
Add to PyInstaller command:
```bash
--add-data "path/to/file:destination"
```

Example:
```bash
--add-data "makedown:makedown" \
--add-data "README.md:."
```

### Console Output (Debugging)
Remove `--windowed` flag:
```bash
pyinstaller --onefile \
    --name pyTools_ResoLab \
    --hidden-import=PyQt5.sip \
    main.py
```

## File Sizes

| Type | Approximate Size |
|------|------------------|
| Single EXE | 200-300 MB |
| Folder Mode | 250-350 MB |
| Source Distribution | 60 KB |

## Requirements on Target Machine

### Windows:
- Windows 7, 8, 10, 11 (64-bit or 32-bit)
- Minimum 500 MB free disk space
- No Python installation required

### macOS:
- Not supported with current PyInstaller setup (see py2app instead)

### Linux:
- Not supported with current PyInstaller setup

## Troubleshooting

### Issue: Missing modules error
**Solution**: Add to hidden-import:
```bash
--hidden-import=module_name
```

### Issue: Slow startup
**Solution**: Use folder mode instead of --onefile

### Issue: PyQt5 not found
**Solution**: Ensure PyQt5 is installed
```bash
pip install PyQt5>=5.15.0
```

### Issue: DLL errors on target machine
**Solution**: Use folder mode (includes all dependencies)

## Building a Release

### Steps:
1. Update version in setup.py and README.md
2. Build the executable:
```bash
pyinstaller --onefile \
    --windowed \
    --name pyTools_ResoLab_v2.0 \
    --add-data "makedown:makedown" \
    --hidden-import=PyQt5.sip \
    main.py
```

3. Zip the dist folder:
```bash
# Windows
Compress-Archive -Path dist/pyTools_ResoLab -DestinationPath pyTools_ResoLab_v2.0_win.zip

# macOS/Linux
zip -r pyTools_ResoLab_v2.0_win.zip dist/pyTools_ResoLab
```

4. Upload to GitHub Releases

## Distributing via GitHub Releases

1. Go to GitHub repository
2. Click "Releases" → "Create a new release"
3. Tag version: `v2.0`
4. Title: "pyTools_ResoLab v2.0 - Windows Release"
5. Upload the .zip file
6. Add release notes

## GitHub Actions (Automated Builds)

To automate Windows builds, create `.github/workflows/build.yml`:

```yaml
name: Build Windows EXE

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt pyinstaller
      - run: pyinstaller --onefile --windowed --name pyTools_ResoLab main.py
      - uses: actions/upload-artifact@v2
        with:
          name: pyTools_ResoLab.exe
          path: dist/
```

## Performance Tips

1. Use `--onefile` for simplicity, folder mode for speed
2. Exclude unnecessary modules with `--exclude-module`
3. Use UPX to compress binaries (optional)
4. Test on target machine before release

## Cleanup

Remove build artifacts:
```bash
rm -rf build/
rm -rf dist/
rm -rf *.spec
```

## Additional Resources

- PyInstaller Documentation: https://pyinstaller.org/
- PyQt5 PyInstaller Guide: https://pyinstaller.org/en/stable/
- Windows Defender SmartScreen: Some users may see warnings (normal for unsigned executables)

---

**Version**: 2.0  
**Last Updated**: December 2, 2025  
**Project**: pyTools_ResoLab

