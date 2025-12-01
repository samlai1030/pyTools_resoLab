# SFR Analyzer with Edge Detection

A professional-grade Spatial Frequency Response (SFR) analyzer built with PyQt5 for analyzing raw camera images.

## Features

### Image Processing
- **Load Raw Images**: Support for uint8, uint16, and float32 data types
- **Mouse Wheel Zoom**: 0.5x to 5.0x zoom with smooth scaling
- **Scrollbars**: Horizontal and vertical navigation for zoomed images
- **ROI Selection**: Click and drag to select regions of interest
- **ROI Preview**: Real-time display of selected region

### Edge Detection
- **Automatic Edge Orientation Detection**:
  - V-Edge (Vertical): Detects vertical edges for horizontal MTF testing
  - H-Edge (Horizontal): Detects horizontal edges for vertical MTF testing
  - Mixed: Identifies unclear or diagonal edges
- **Confidence Scoring**: 0-100% confidence metrics for detection reliability
- **Edge Validation**: Automatic validation of edge quality and contrast

### SFR Analysis
- **Adaptive Calculation**: Different methods for V-Edge and H-Edge
- **MTF Computation**: Calculates Modulation Transfer Function
- **MTF50 Extraction**: Automatically finds the 50% MTF frequency
- **FFT-based**: Uses Fourier transform for accurate frequency response

### User Interface
- **Professional Layout**: Balanced image and plot display
- **Real-time Status**: Shows edge type, confidence, and MTF50 values
- **Interactive Plotting**: matplotlib integration for SFR visualization
- **Responsive UI**: Smooth, lag-free interaction

## Requirements

```
PyQt5>=5.15
numpy>=1.20
scipy>=1.7
opencv-python>=4.5
matplotlib>=3.3
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SFR-analyzer.git
cd SFR-analyzer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python SFR_app_v2.py
```

### Workflow
1. Click "Load .raw File" to load a raw image
2. Enter image dimensions (width, height) and data type
3. Use mouse wheel to zoom in/out
4. Use scrollbars to navigate zoomed areas
5. Click and drag to select ROI (Region of Interest)
6. View ROI preview below the main image
7. Automatic edge detection and SFR calculation
8. Results displayed with MTF plot and metrics

### Testing
```bash
python test_edge_detection.py
```

Runs comprehensive tests for:
- Edge detection (V-Edge, H-Edge, Mixed)
- ROI validation
- SFR calculation
- Performance benchmarks

## Key Technical Features

### Edge Detection Algorithm
- Uses Sobel gradient operators
- Calculates magnitude ratio (X/Y gradients)
- Classifies edges based on gradient distribution
- Provides confidence scoring

### SFR Calculation
- Extracts Edge Spread Function (ESF)
- Computes Line Spread Function (LSF)
- Applies Hamming window for spectral smoothing
- FFT transformation for frequency response
- Normalization to DC component

### Zoom & Selection
- Smooth image scaling with Qt.SmoothTransformation
- Scrollbar integration for navigation
- Accurate coordinate conversion between zoomed and original images
- Real-time ROI preview display

## File Structure

```
SFR-analyzer/
├── SFR_app_v2.py              # Main application (PyQt5)
├── SFR_app_v2_PyQt5.py        # Alternative PyQt5 version
├── test_edge_detection.py     # Comprehensive test suite
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Architecture

### Main Components

**ImageLabel**
- Custom QLabel for image display
- Handles mouse wheel zoom events
- Supports ROI selection with visual feedback
- Manages coordinate conversion

**SFRCalculator**
- Static methods for edge detection and SFR calculation
- `detect_edge_orientation()`: Identifies edge type
- `validate_edge()`: Validates edge quality
- `calculate_sfr()`: Computes frequency response

**MainWindow**
- PyQt5 main application window
- Layout management for image and plot areas
- ROI processing and analysis workflow
- Results visualization

## Performance

- Edge detection: ~5-10 ms
- SFR calculation: ~1-2 ms
- Total analysis latency: ~350 ms
- Smooth zoom and scroll interaction
- Responsive GUI with no lag

## Testing Results

All 14 comprehensive tests pass:
- ✅ V-Edge detection (100% accuracy)
- ✅ H-Edge detection (100% accuracy)
- ✅ Mixed edge detection
- ✅ Edge validation
- ✅ SFR calculation
- ✅ Performance benchmarks

## Future Enhancements

- [ ] Slanted edge support with angle measurement
- [ ] Batch ROI processing
- [ ] User-configurable detection thresholds
- [ ] Export results to CSV/PDF
- [ ] Advanced visualization options

## License

MIT License

## Author

Sam Lai

## Support

For issues, questions, or contributions, please create an issue or submit a pull request on GitHub.

---

**Status**: Production Ready ✅
**Version**: 1.0
**Last Updated**: November 29, 2025

