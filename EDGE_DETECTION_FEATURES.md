# Edge Detection Features - V-Edge and H-Edge

## Overview

Both `SFR_app_v2.py` and `SFR_app_v2_PyQt5.py` now include advanced edge orientation detection that automatically identifies whether your cropped ROI contains a **Vertical Edge (V-Edge)** or **Horizontal Edge (H-Edge)**.

## Features Added

### 1. **Edge Orientation Detection**

A new method `detect_edge_orientation()` analyzes the cropped image to determine:

- **Edge Type**: V-Edge (垂直邊), H-Edge (水平邊), or Mixed (混合邊)
- **Confidence Score**: 0-100% indicating how confident the detection is
- **Detailed Metrics**: Gradient magnitudes, ratios, and angle distributions

#### Detection Algorithm:

1. **Sobel Gradient Calculation**
   - `sobelx`: Detects vertical edges (x-direction gradients)
   - `sobely`: Detects horizontal edges (y-direction gradients)

2. **Gradient Analysis**
   - Computes total magnitude in both directions
   - Calculates gradient direction angles (0-180°)
   - Counts pixels with dominant directions

3. **Edge Classification**
   ```
   - V-Edge (Vertical):     mag_x / mag_y > 1.5
   - H-Edge (Horizontal):   mag_y / mag_x > 1.5
   - Mixed:                 1.5 < ratio < 0.67
   ```

4. **Confidence Scoring**
   - Higher confidence for more pronounced directional gradients
   - Scaled from 0-100%

### 2. **Adaptive SFR Calculation**

The `calculate_sfr()` method now adapts based on edge type:

**For V-Edge (Vertical Edge):**
```python
esf = np.mean(img, axis=0)  # Column average (ESF along x-axis)
```
- Calculates Edge Spread Function along the horizontal direction
- Performs FFT on the LSF (vertical profile differential)

**For H-Edge (Horizontal Edge):**
```python
esf = np.mean(img, axis=1)  # Row average (ESF along y-axis)
```
- Calculates Edge Spread Function along the vertical direction
- Performs FFT on the LSF (horizontal profile differential)

### 3. **User Interface Improvements**

#### Information Display:
- **Status Bar** shows detected edge type and confidence level
- **Plot Title** indicates which edge type was analyzed
- **MTF50 Calculation** displayed with edge type context

#### Example Output:
```
"V-Edge (Confidence: 87.5%) | MTF50: 0.234 cy/px | SFR Calculated"
"H-Edge (Confidence: 92.3%) | MTF50: 0.189 cy/px | SFR Calculated"
```

## How to Use

### Step 1: Load Raw Image
- Click "Load .raw File"
- Specify Width, Height, and Data Type (uint8, uint16, float32)
- Image will be displayed

### Step 2: Select ROI
- **Click and drag** on the image to select your crop region
- Red dashed rectangle shows your selection
- Release mouse to process

### Step 3: Automatic Analysis
The app automatically:
1. ✅ Detects if ROI contains an edge (contrast check)
2. ✅ Identifies edge orientation (V-Edge or H-Edge)
3. ✅ Calculates SFR with appropriate method
4. ✅ Displays results with confidence score

## Technical Details

### Edge Detection Metrics

The system tracks these metrics for each ROI:

| Metric | Description |
|--------|-------------|
| `mag_x` | Total x-direction gradient magnitude |
| `mag_y` | Total y-direction gradient magnitude |
| `ratio_x_y` | Magnitude ratio (mag_x / mag_y) |
| `v_edges_percent` | Percentage of pixels with ~90° angle (vertical) |
| `h_edges_percent` | Percentage of pixels with ~0°/180° angle (horizontal) |
| `mean_x` | Average x-direction gradient intensity |
| `mean_y` | Average y-direction gradient intensity |

### Validation Thresholds

| Check | Threshold | Description |
|-------|-----------|-------------|
| Edge Detection | max(magnitude) ≥ 50 | Minimum contrast requirement |
| V-Edge Classification | ratio > 1.5 | X-gradient at least 1.5x stronger |
| H-Edge Classification | ratio < 0.67 | Y-gradient at least 1.5x stronger |
| Confidence Calculation | 0-100% | Scaled based on magnitude ratio |

## Example Workflows

### Workflow 1: Analyzing Vertical Edge
```
1. Load raw image (1920×1080, uint16)
2. Crop a region containing a vertical line/edge
3. App detects "V-Edge, Confidence: 88%"
4. SFR calculated using column averaging
5. MTF50 ≈ 0.25 cy/px
```

### Workflow 2: Analyzing Horizontal Edge
```
1. Load raw image (1920×1080, uint16)
2. Crop a region containing a horizontal line/edge
3. App detects "H-Edge, Confidence: 91%"
4. SFR calculated using row averaging
5. MTF50 ≈ 0.22 cy/px
```

## Physical Interpretation

### Why Edge Orientation Matters:

- **Vertical Edge (V-Edge)**: 
  - Tests MTF along the horizontal axis (column direction)
  - Useful for measuring horizontal resolution
  - Common in target patterns (vertical slits)

- **Horizontal Edge (H-Edge)**:
  - Tests MTF along the vertical axis (row direction)
  - Useful for measuring vertical resolution
  - Common in target patterns (horizontal slits)

- **Mixed Edge**:
  - Indicates slanted or unclear edges
  - Less ideal for precise SFR measurement
  - Confidence score will be moderate (~50%)

## Implementation Notes

### File Structure:
```
SFRCalculator (class)
├── detect_edge_orientation() - New method
├── validate_edge()           - Updated (now returns edge type & confidence)
└── calculate_sfr()           - Updated (handles V and H edges)

ImageLabel (class)
└── roi_callback              - Improved with callback pattern

MainWindow (class)
├── process_roi()    - Updated (uses edge detection)
├── plot_sfr()       - Updated (shows edge type)
└── load_raw_file()  - Unchanged
```

### Dependencies:
- OpenCV (`cv2.Sobel()`) for gradient calculation
- NumPy for angle computation
- SciPy (`scipy.fftpack`) for FFT
- PyQt5 for GUI

## Troubleshooting

### Issue: "Detection Failed: Low Contrast"
- **Cause**: ROI lacks clear edge (too uniform or noisy)
- **Solution**: Select a region with sharper edge contrast

### Issue: "Mixed Edge" with low confidence
- **Cause**: Edge is diagonal/slanted or unclear
- **Solution**: Try to select a more vertical or horizontal edge

### Issue: High confidence but poor MTF curve
- **Cause**: Edge may be too thick or have sub-pixel features
- **Solution**: Select a smaller, cleaner ROI region

## Future Enhancements

Potential improvements:
- [ ] Sub-pixel edge alignment for slanted edges
- [ ] Automatic ROI suggestion based on image analysis
- [ ] Multi-point edge detection for composite analysis
- [ ] Edge angle/slant measurement
- [ ] Batch processing multiple ROIs

---

**Version**: 1.0
**Last Updated**: November 26, 2025
**Status**: Production Ready ✓

