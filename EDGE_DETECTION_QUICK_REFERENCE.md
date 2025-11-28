# Quick Reference - V-Edge vs H-Edge Detection

## At a Glance

| Feature | V-Edge | H-Edge | Mixed |
|---------|--------|--------|-------|
| **Edge Orientation** | Vertical (⏐) | Horizontal (─) | Diagonal/Unclear (/) |
| **Gradient Ratio** | X > Y (1.5×) | Y > X (1.5×) | Balanced |
| **ESF Direction** | Horizontal (row→col) | Vertical (col→row) | Both |
| **MTF Tests** | Horizontal resolution | Vertical resolution | Unreliable |
| **Ideal For** | Vertical slits/lines | Horizontal slits/lines | Not recommended |

## Detection Process Flow

```
┌─────────────────────────────────────┐
│   User Selects ROI on Image        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  validate_edge() checks:             │
│  ✓ Is ROI empty?                    │
│  ✓ Is contrast sufficient?          │
│  → Calls detect_edge_orientation()  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  detect_edge_orientation() analyzes: │
│  • Sobel X and Y gradients          │
│  • Magnitude ratio (X/Y)            │
│  • Angle histogram                  │
└────────────┬────────────────────────┘
             │
      ┌──────┴──────┬──────────┐
      │             │          │
      ▼             ▼          ▼
   V-Edge       H-Edge      Mixed
   (X>1.5Y)     (Y>1.5X)    (Balanced)
      │             │          │
      └──────┬──────┴──────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  calculate_sfr() adapts method:     │
│  V-Edge: esf = mean(img, axis=0)   │
│  H-Edge: esf = mean(img, axis=1)   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Display Results:                    │
│  • Edge type + Confidence            │
│  • MTF50 value                       │
│  • SFR curve plot                    │
└─────────────────────────────────────┘
```

## Key Code Snippets

### Detecting Edge Type:
```python
edge_type, confidence, details = SFRCalculator.detect_edge_orientation(roi)

# Possible values:
# edge_type: "V-Edge", "H-Edge", "Mixed", or "No Edge"
# confidence: 0-100 (float)
# details: dict with mag_x, mag_y, ratio_x_y, etc.
```

### Calculating SFR with Edge Type:
```python
frequencies, mtf = SFRCalculator.calculate_sfr(roi, edge_type="V-Edge")
# or
frequencies, mtf = SFRCalculator.calculate_sfr(roi, edge_type="H-Edge")
```

### Getting Full Validation:
```python
is_valid, msg, edge_type, confidence = SFRCalculator.validate_edge(roi)

if is_valid:
    print(f"{edge_type} detected with {confidence:.1f}% confidence")
else:
    print(f"Validation failed: {msg}")
```

## Confidence Score Interpretation

| Confidence | Reliability | Recommendation |
|------------|------------|-----------------|
| > 90% | Excellent | Use results with confidence |
| 80-90% | Very Good | Results are reliable |
| 70-80% | Good | Acceptable, check visually |
| 60-70% | Fair | Consider re-cropping |
| 50-60% | Poor | Edge likely mixed/unclear |
| < 50% | Very Poor | Recommend reselection |

## Physical Interpretation

### For Image Sensors:

**V-Edge Measurement** → Horizontal MTF
- Tests sharpness in left-right direction
- Evaluates pixel columns' resolving power
- Common in pattern targets (vertical test pattern)

**H-Edge Measurement** → Vertical MTF
- Tests sharpness in up-down direction
- Evaluates pixel rows' resolving power
- Common in pattern targets (horizontal test pattern)

## Troubleshooting Checklist

- [ ] Is the ROI large enough? (min 5×5 pixels)
- [ ] Does ROI contain a clear, visible edge?
- [ ] Is contrast sufficient? (not too uniform)
- [ ] Is edge mostly vertical or mostly horizontal?
- [ ] Is the edge straight (not curved)?
- [ ] Are there artifacts at ROI boundaries?

## Common Scenarios

### ✓ Good V-Edge Crop
```
Clean vertical line separating dark/light regions
Sufficient contrast
Rectangular selection spanning the edge
```

### ✓ Good H-Edge Crop
```
Clean horizontal line separating dark/light regions
Sufficient contrast
Rectangular selection spanning the edge
```

### ✗ Bad Crop - Mixed Signal
```
Diagonal edge
Very thin line
Multiple edges in ROI
Low contrast or noisy region
```

---

**Quick Start**: Load image → Drag to select edge region → App auto-detects type → Check MTF50 value

