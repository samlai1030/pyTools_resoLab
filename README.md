# pyTools_ResoLab - Professional SFR/MTF Analyzer

A professional-grade Spatial Frequency Response (SFR) / Modulation Transfer Function (MTF) analyzer built with PyQt5, following **ISO 12233:2023** standard for analyzing raw camera images.

## 🎯 Project Status

**Status**: Production Ready ✅
**Version**: 2.4
**Last Updated**: December 5, 2025
**GitHub**: https://github.com/samlai1030/pyTools_ResoLab

---

## ✨ Features

### 📁 Image Loading
- **RAW Image Support**: uint8, uint16, float32 data types
- **Auto-Detection**: Automatic width/height detection from filename and file size
- **Recent Files**: Quick access to last 10 opened files via combo box
- **Persistent History**: Recent files saved between sessions

### 🎯 ROI Selection Modes
| Mode | Description |
|------|-------------|
| **Drag Select** | Click and drag to select any custom ROI size |
| **Click Select** | Single click to select fixed-size area (default 40×40, configurable 10-200) |

### 👁️ View Modes
| Mode | Description |
|------|-------------|
| **📊 SFR** | Select ROI for SFR/MTF analysis |
| **🖐 VIEW** | Pan/scroll image with mouse drag |

### 🔍 Edge Detection
- **Automatic Orientation Detection**: V-Edge (Vertical) / H-Edge (Horizontal)
- **Confidence Scoring**: 0-100% reliability metrics
- **Adjustable Threshold**: Slider control (10-200)
- **Canny Edge Overlay**: Visual edge preview on image
- **Apply/Erase Edge**: Lock or clear edge pattern as reference

### 📈 SFR Analysis (ISO 12233:2023 Compliant)
- **4× Supersampling**: Improved accuracy with frequency compensation
- **ESF Calculation**: Edge Spread Function
- **LSF Calculation**: Line Spread Function with FWHM display
- **FFT-based MTF**: Accurate frequency response computation
- **White Area Compensation**: Bias and noise correction
- **Ny/4 Reference**: Nyquist quarter frequency marker with SFR value

### 🔧 LSF Smoothing Methods
| Method | Description |
|--------|-------------|
| `none` | No smoothing (default) |
| `savgol` | Savitzky-Golay filter |
| `gaussian` | Gaussian smoothing |
| `median` | Median filter |
| `uniform` | Uniform/box filter |
| `butterworth` | Butterworth IIR filter |
| `wiener` | Wiener adaptive filter |

### ⚡ SFR Stabilization
- **3-Sample Averaging**: Optional multi-sample averaging for stable measurements
- **Noise Reduction**: Automatic noise suppression
- **Stability Metrics**: Standard deviation display

### 🎚️ Nyquist Frequency Control
- **Adjustable Range**: 0.10 to 1.00 via slider
- **Real-time Update**: SFR plot updates instantly when slider changes
- **Ny/4 Reference Line**: Green dashed line on SFR plot

### 📊 Visualization (2×2 Subplot Layout)
| Position | Content |
|----------|---------|
| Top-Left | **SFR/MTF Curve** with Ny/4 reference line |
| Top-Right | **ROI Image** preview with dimensions |
| Bottom-Left | **ESF** (Edge Spread Function) |
| Bottom-Right | **LSF** (Line Spread Function) with FWHM |

### 🖥️ UI Features
- **Resizable Panels**: Splitter control between image and plot areas
- **Group Boxes**: Organized controls (File, Selection Mode, Image View, SFR Result, Nyquist)
- **Mouse Wheel Zoom**: 0.5x to 5.0x with smooth scaling
- **Status Display**: Real-time edge type, confidence, MTF50, and SFR values
- **SFR Overlay**: Value displayed at ROI position on image

---

## 📋 Requirements

```
PyQt5>=5.15.0
numpy>=1.21.0
scipy>=1.7.0
opencv-python>=4.5.0
matplotlib>=3.3.0
pillow>=10.0.0
```

---

## 📦 Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install pyTools-ResoLab
```

### Option 2: Install from GitHub

```bash
pip install git+https://github.com/samlai1030/pyTools_ResoLab.git
```

### Option 3: Clone and Install

```bash
# Clone the repository
git clone https://github.com/samlai1030/pyTools_ResoLab.git
cd pyTools_ResoLab

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Option 4: Download macOS App

Download the pre-built `.dmg` from [Releases](https://github.com/samlai1030/pyTools_ResoLab/releases)

---

## 🚀 Usage

### Basic Workflow

1. **Load Image**: Click `Load .raw` or select from Recent Files
2. **Set Dimensions**: Enter width, height, and data type (if not auto-detected)
3. **Select Mode**: Choose `Drag Select` or `Click` for ROI selection
4. **Select ROI**: Click/drag on a slant edge in the image
5. **View Results**:
   - 2×2 plot displays ESF, LSF, SFR, and ROI image
   - Status bar shows edge type, confidence, MTF50, SFR@Ny/4
6. **Adjust Settings**:
   - LSF smoothing method
   - Nyquist frequency slider
   - SFR Stabilize checkbox

### Keyboard & Mouse Controls

| Action | Control |
|--------|---------|
| Zoom In/Out | Mouse Wheel |
| Pan Image | Right-click drag (or VIEW mode + left-click) |
| Select ROI | Left-click drag (SFR mode) |
| Quick Select | Left-click (Click mode) |

---

## 🏗️ Project Structure

```
pyTools_ResoLab/
├── main.py                 # Main application entry point (MainWindow class)
├── mainUI.py               # Generated UI code (from pyuic5)
├── mainUI.ui               # Qt Designer UI file
│
├── constants.py            # Application constants and configuration
│   ├── SUPERSAMPLING_FACTOR, EPSILON, MAX_FILE_SIZE_MB
│   ├── RAW_FORMAT_OPTIONS  # UI dropdown options for raw formats
│   └── COMMON_RAW_SIZES    # Auto-detection dimension list
│
├── utils.py                # Utility functions for file I/O
│   ├── read_raw_image()    # Read raw image with validation
│   ├── remove_inactive_borders()  # Crop black borders
│   └── auto_detect_raw_dimensions()  # Auto-detect W×H from file
│
├── sfr_calculator.py       # ISO 12233:2023 SFR/MTF computation
│   ├── _apply_lsf_smoothing()  # LSF smoothing methods
│   ├── detect_edge_orientation()  # V-Edge / H-Edge detection
│   ├── validate_edge()     # Edge validation with threshold
│   ├── standardize_roi_orientation()  # ROI orientation normalization
│   └── calculate_sfr()     # Main SFR calculation algorithm
│
├── image_label.py          # Custom QLabel widget for image display
│   ├── Mouse event handlers (drag/click/pan)
│   ├── Zoom with mouse wheel (0.5x - 5.0x)
│   ├── ROI selection (drag rectangle / click square)
│   └── ROI markers display with SFR values
│
├── requirements.txt        # Python dependencies
├── recent_files.json       # Persistent recent files list
├── recent_roi_files.json   # Persistent recent ROI files list
└── README.md               # This file
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| `MainWindow` | `main.py` | Main application window with UI controls and event handling |
| `SFRCalculator` | `sfr_calculator.py` | Static class for ISO 12233:2023 SFR/MTF computation |
| `ImageLabel` | `image_label.py` | Custom QLabel with zoom, pan, ROI selection |
| `Ui_MainWindow` | `mainUI.py` | Auto-generated UI from Qt Designer |
| Constants | `constants.py` | Application-wide constants and format options |
| Utilities | `utils.py` | File I/O and image preprocessing functions |

---

## 📊 Features Matrix

| Feature | Status | Description |
|---------|--------|-------------|
| RAW Image Loading | ✅ | uint8, uint16, float32 support |
| Auto W/H Detection | ✅ | From filename and file size |
| Recent Files | ✅ | Persistent combo box (10 files) |
| Drag/Click Selection | ✅ | Dual ROI selection modes |
| SFR/VIEW Modes | ✅ | Analysis and pan modes |
| Edge Detection | ✅ | V-Edge, H-Edge with confidence |
| ISO 12233:2023 SFR | ✅ | 4× supersampling, FFT-based |
| LSF Smoothing | ✅ | 7 methods including none |
| SFR Stabilize | ✅ | 3-sample averaging |
| Nyquist Slider | ✅ | Real-time plot update |
| 2×2 Plot Layout | ✅ | SFR, ROI, ESF, LSF |
| Resizable Panels | ✅ | Splitter control |
| FWHM Display | ✅ | On LSF plot |

---

## 🔧 Building Executable

### macOS (.app)

```bash
# Install PyInstaller
pip install pyinstaller

# Build with spec file (includes recursion limit fix)
pyinstaller pyTools_ResoLab.spec

# Output: dist/pyTools_ResoLab
```

### Create DMG

```bash
cd dist
hdiutil create -volname "pyTools_ResoLab" -srcfolder . -ov -format UDZO pyTools_ResoLab_v2.4.dmg
```

---

## 📝 Changelog

### v2.4 (December 5, 2025)
- UI layout reorganization with group boxes
- Resizable left/right panels via splitter
- Nyquist control moved to bottom with dedicated group box
- SFR Result group box with compact horizontal layout
- Edge Detection moved under Image View
- LSF default filter set to "none"
- Removed fixed figure size for flexible plot resizing

### v2.0 (December 2, 2025)
- Initial production release
- ISO 12233:2023 compliant SFR calculation
- 2×2 subplot layout with ROI image
- Recent files persistence
- Auto W/H detection

---

## 📄 License

MIT License

---

## 👤 Author

**Sam Lai**
GitHub: [@samlai1030](https://github.com/samlai1030)
