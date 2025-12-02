#!/bin/bash
# Build script for packaging pyTools_ResoLab as Windows .exe

echo "================================"
echo "pyTools_ResoLab - EXE Builder"
echo "================================"
echo ""

# Check if PyInstaller is installed
echo "Checking for PyInstaller..."
pip show pyinstaller > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
else
    echo "PyInstaller is already installed."
fi

echo ""
echo "Building executable..."
echo "This may take a few minutes..."
echo ""

# Build the executable
pyinstaller --onefile \
    --windowed \
    --name pyTools_ResoLab \
    --add-data "makedown:makedown" \
    --hidden-import=PyQt5.sip \
    SFR_app_v2.py

echo ""
echo "================================"
echo "Build Complete!"
echo "================================"
echo ""
echo "Output location:"
echo "  - Single file EXE: ./dist/pyTools_ResoLab.exe"
echo "  - Full folder: ./dist/pyTools_ResoLab/"
echo ""
echo "To use the executable:"
echo "  1. Copy the .exe file or entire dist folder"
echo "  2. Run pyTools_ResoLab.exe"
echo "  3. No Python installation required on target machine"
echo ""
echo "Note: First run may be slow due to library loading"
echo ""

