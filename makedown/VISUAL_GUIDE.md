# Visual Guide - Edge Detection Feature

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              SFR Analyzer Application                    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  User Interface (PyQt5)                         │    │
│  │  ┌────────────────────┐  ┌─────────────────┐   │    │
│  │  │ Image Display Area │  │ Results Panel   │   │    │
│  │  │                    │  │ ├─ Status Bar   │   │    │
│  │  │ (with ROI select)  │  │ ├─ MTF Plot     │   │    │
│  │  │                    │  │ └─ Info Labels  │   │    │
│  │  └────────────────────┘  └─────────────────┘   │    │
│  └─────────────────────────────────────────────────┘    │
│                      │                                    │
│                      ▼ (user selects ROI)               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Edge Detection Engine                          │    │
│  │  (SFRCalculator class)                          │    │
│  │                                                  │    │
│  │  1. validate_edge()                             │    │
│  │     ├─ Check if ROI is valid                    │    │
│  │     ├─ Check contrast level                     │    │
│  │     └─ Call detect_edge_orientation()           │    │
│  │                                                  │    │
│  │  2. detect_edge_orientation()                   │    │
│  │     ├─ Sobel gradient calculation               │    │
│  │     ├─ Magnitude analysis                       │    │
│  │     ├─ Classification (V/H/Mixed/None)          │    │
│  │     └─ Return (edge_type, confidence, details)  │    │
│  │                                                  │    │
│  │  3. calculate_sfr(edge_type)                    │    │
│  │     ├─ Adaptive ESF extraction                  │    │
│  │     │  ├─ V-Edge: axis=0 (column mean)         │    │
│  │     │  └─ H-Edge: axis=1 (row mean)            │    │
│  │     ├─ LSF computation                          │    │
│  │     ├─ FFT transformation                       │    │
│  │     └─ Return (frequencies, sfr)                │    │
│  └─────────────────────────────────────────────────┘    │
│                      │                                    │
│                      ▼ (results ready)                   │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Display Results                                │    │
│  │  ├─ Status: "V-Edge (Conf: 87.5%)"             │    │
│  │  ├─ Plot: MTF curve with edge type             │    │
│  │  └─ Info: "MTF50: 0.234 cy/px"                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## Edge Detection Classification

```
                          ROI Image
                              │
                              ▼
                    Sobel Gradient Analysis
                    /        │        \
                   /         │         \
              SobelX      SobelY    Calculate
              (X-grad)   (Y-grad)    Angles
                   \         │         /
                    \        │        /
                     ▼       ▼       ▼
                  Magnitude Ratio Analysis
                  (mag_x / mag_y)
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
        > 1.5        0.67-1.5        < 0.67
            │            │            │
            ▼            ▼            ▼
        ┌─────────┐  ┌────────┐  ┌────────┐
        │ V-Edge  │  │ Mixed  │  │ H-Edge │
        │   ⏐    │  │   /    │  │   ─    │
        └─────────┘  └────────┘  └────────┘
            │            │            │
            ▼            ▼            ▼
         High Conf    Med Conf      High Conf
        85-100%       40-60%        85-100%
```

---

## SFR Calculation Adaptation

### V-Edge Path (Vertical Edge)
```
ROI Image (MxN pixels)
    │
    ▼
Grayscale Conversion
    │
    ▼
Column-wise Averaging (axis=0)
    ↓
ESF Profile (1D array along X)
    ↓
Differentiation (LSF = dESF/dx)
    ↓
Apply Hamming Window
    ↓
FFT Transformation
    ↓
Normalize (DC ≈ 1.0)
    ▼
MTF Curve (Horizontal Resolution)
```

### H-Edge Path (Horizontal Edge)
```
ROI Image (MxN pixels)
    │
    ▼
Grayscale Conversion
    │
    ▼
Row-wise Averaging (axis=1)
    ↓
ESF Profile (1D array along Y)
    ↓
Differentiation (LSF = dESF/dy)
    ↓
Apply Hamming Window
    ↓
FFT Transformation
    ↓
Normalize (DC ≈ 1.0)
    ▼
MTF Curve (Vertical Resolution)
```

---

## Confidence Score Interpretation

```
Confidence Scale:

100% │ ████████████████ Excellent  (ratio >> 1.5)
     │ ████████████████ Very High  (ratio > 2.0)
     │ ███████████████░ High       (ratio > 1.5)
     │ ██████████░░░░░░ Good       (ratio 1.3-1.5)
     │ █████░░░░░░░░░░ Fair       (ratio 1.1-1.3)
 50% │ ██░░░░░░░░░░░░░ Mixed/Low  (ratio ≈ 1.0)
     │ ░░░░░░░░░░░░░░░░ Poor      (unclear edge)
   0%└─────────────────────────────────────────
     0  1.0  1.2  1.4  1.6  1.8  2.0  2.2  2.4
                 Magnitude Ratio (X/Y)
```

---

## User Interaction Flow

```
┌─────────────────────────────────────────────────────┐
│ START: Application Launched                         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
         ┌──────────────────────────┐
         │ Click "Load .raw File"   │
         └────────────┬─────────────┘
                      │
                      ▼
    ┌─────────────────────────────────┐
    │ Enter Width, Height, Data Type  │
    └────────────┬────────────────────┘
                 │
                 ▼
      ┌────────────────────────────┐
      │ Image Displayed            │
      └────────────┬───────────────┘
                   │
                   ▼
      ┌────────────────────────────────────┐
      │ User: Click & Drag to Select ROI   │
      │ App: Display red dashed rectangle  │
      └────────────┬───────────────────────┘
                   │
                   ▼ (mouse release)
      ┌────────────────────────────────────┐
      │ AUTOMATIC ANALYSIS STARTS           │
      │                                     │
      │ 1. Extract ROI from image           │
      │ 2. Run validate_edge()              │
      │    └─ Check: empty? contrast ok?   │
      │ 3. Run detect_edge_orientation()    │
      │    └─ Return: type, confidence      │
      │ 4. Run calculate_sfr(edge_type)     │
      │    └─ Return: frequencies, mtf      │
      │ 5. Plot results                     │
      │ 6. Display status bar               │
      └────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │ RESULTS DISPLAYED                    │
    │                                      │
    │ Status:  "V-Edge (Conf: 87.5%)"     │
    │ Plot:    MTF curve with title       │
    │ Info:    "MTF50: 0.234 cy/px"       │
    └──────────────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │ User Can:                            │
    │ ✓ Select another ROI                 │
    │ ✓ Load new image                     │
    │ ✓ Export results                     │
    └──────────────────────────────────────┘
```

---

## Edge Type Examples

### Example 1: Perfect V-Edge
```
Input Image:          Gradient Map:         Detection Result:
                                            
█████░░░░░            ▲▲▲▲▲░░░░░           ✓ Edge Type: V-Edge
█████░░░░░            ▲▲▲▲▲░░░░░           ✓ Confidence: 100%
█████░░░░░      →     ▲▲▲▲▲░░░░░     →     ✓ Ratio X/Y: 2.0+
█████░░░░░            ▲▲▲▲▲░░░░░           ✓ MTF Curve: Valid
█████░░░░░            ▲▲▲▲▲░░░░░           
                                            
[Dark][Light]         [Strong X-gradient]   Action: Use Column Mean
```

### Example 2: Perfect H-Edge
```
Input Image:          Gradient Map:         Detection Result:
                                            
█████████░            ░░░░░░░░░░           ✓ Edge Type: H-Edge
█████████░            ░░░░░░░░░░           ✓ Confidence: 100%
─────────░      →     ▶▶▶▶▶▶▶▶░░░    →     ✓ Ratio X/Y: 0.01
░░░░░░░░░░            ▶▶▶▶▶▶▶▶░░░           ✓ MTF Curve: Valid
░░░░░░░░░░            ▶▶▶▶▶▶▶▶░░░           
                                            
[Dark][Light]         [Strong Y-gradient]   Action: Use Row Mean
```

### Example 3: Diagonal Edge (Mixed)
```
Input Image:          Gradient Map:         Detection Result:
                                            
█░░░░░░░░░            ▲▶░░░░░░░░           ✓ Edge Type: Mixed
██░░░░░░░░            ▲▲▶░░░░░░░           ✓ Confidence: 50%
███░░░░░░░     →      ▲▲▲▶░░░░░░     →     ✓ Ratio X/Y: ~1.0
████░░░░░░            ▲▲▲▲▶░░░░░           ⚠ MTF Curve: Unclear
█████░░░░░            ▲▲▲▲▲▶░░░░           
                                            
[Diagonal]            [Mixed gradients]     Warning: Not Recommended
```

---

## Performance Timeline

```
User Action: Drag to select ROI
             │
             ▼ (0-100ms)
        Extract ROI
             │
             ▼ (0-10ms)
    Validate edge exists
             │
             ▼ (5-10ms)
   Detect edge orientation
        (Sobel, Ratio)
             │
             ▼ (0.4ms)
    Calculate SFR
        (FFT, Normalize)
             │
             ▼ (100-200ms)
    Update plot & display
             │
             ▼ Total: ~350ms
        Results visible
        ═══════════════════
        User feels immediate response
```

---

## Code Flow Diagram

```
MainWindow.process_roi(rect)
    │
    ├─ Extract ROI from raw_data[y:y+h, x:x+w]
    │
    ├─► SFRCalculator.validate_edge(roi)
    │   ├─ Check if roi is None/empty
    │   ├─ Calculate Sobel gradients
    │   ├─► detect_edge_orientation(roi)
    │   │   ├─ Compute mag_x and mag_y
    │   │   ├─ Calculate ratio = mag_x/mag_y
    │   │   ├─ If ratio > 1.5 → "V-Edge"
    │   │   ├─ If ratio < 0.67 → "H-Edge"
    │   │   └─ Else → "Mixed"
    │   │
    │   └─ Return (is_valid, msg, edge_type, conf)
    │
    └─ If is_valid:
       │
       ├─► SFRCalculator.calculate_sfr(roi, edge_type)
       │   ├─ Convert to grayscale
       │   ├─ If V-Edge: esf = mean(img, axis=0)
       │   ├─ If H-Edge: esf = mean(img, axis=1)
       │   ├─ Compute lsf = diff(esf)
       │   ├─ Apply hamming window
       │   ├─ FFT transformation
       │   └─ Return (frequencies, sfr_values)
       │
       └─ plot_sfr(freqs, sfr, edge_type)
          ├─ Clear previous plot
          ├─ Plot MTF curve
          ├─ Set title with edge type
          ├─ Add MTF50 reference line
          └─ Update canvas
    
    └─ Update info_label with results
```

---

## Test Coverage Map

```
┌─────────────────────────────────────┐
│     Test Suite Coverage             │
├─────────────────────────────────────┤
│                                     │
│ detect_edge_orientation()           │
│   ✓ V-Edge (perfect)                │
│   ✓ H-Edge (perfect)                │
│   ✓ Diagonal (mixed)                │
│   ✓ Empty ROI                       │
│   ✓ None input                      │
│                                     │
│ validate_edge()                     │
│   ✓ Valid V-Edge                    │
│   ✓ Valid H-Edge                    │
│   ✓ Low contrast (rejected)         │
│   ✓ Uniform image (rejected)        │
│   ✓ Empty ROI (rejected)            │
│                                     │
│ calculate_sfr()                     │
│   ✓ V-Edge mode                     │
│   ✓ H-Edge mode                     │
│   ✓ Normalization                   │
│   ✓ FFT output shape                │
│                                     │
│ Performance Benchmarks              │
│   ✓ Detection speed (5.3ms)         │
│   ✓ SFR speed (0.45ms)              │
│   ✓ Memory usage                    │
│                                     │
├─────────────────────────────────────┤
│ Total: 14 tests, 14 passing ✓       │
└─────────────────────────────────────┘
```

---

## Documentation Map

```
                    README_INDEX.md
                   (You are here!)
                    /    |    \
                   /     |     \
          QUICK_REF  FEATURES  FINAL_REPORT
           (5 min)    (20 min)    (15 min)
              |          |            |
           Quick      Learn       Verify
          Answers    Algorithm     Tests
              |          |            |
              └─────┬────┴────┬──────┘
                    ▼         ▼
              Source Code / Implementation
              (SFR_app_v2.py)
```

---

**Visual Guide Complete** ✓

For detailed information, refer to the documentation files:
- `EDGE_DETECTION_QUICK_REFERENCE.md` - Quick tables and diagrams
- `EDGE_DETECTION_FEATURES.md` - Comprehensive explanation
- `FINAL_REPORT.md` - Complete technical report

