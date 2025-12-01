# Click Select Size Input Box - User-Configurable Selection Size

## Implementation Complete ✅

Successfully added a **QSpinBox input control** next to the radio buttons to allow users to dynamically set the click select mode area size.

---

## UI Component Details

### Location
**Selection Mode section**, on the right side of radio buttons

### Layout Structure
```
Selection Mode:
[⦿ Drag Select] [⦾ Click (40×40)] [Size:] [Input Box] [Stretch]
                                    ↑       ↑
                                  Label   Spinner
```

### Components

1. **Label**: "Size:"
   - Font: 10px, bold
   - Width: Auto

2. **Input Box (QSpinBox)**: `self.click_size_input`
   - Minimum value: 10 pixels
   - Maximum value: 200 pixels
   - Default value: 40 pixels
   - Minimum width: 50px
   - Font size: 10px

---

## Features

✅ **User-Configurable**: Set any size from 10 to 200 pixels
✅ **Real-time Update**: Radio button label updates immediately
✅ **Validation**: Min/max bounds enforced (10-200)
✅ **Status Feedback**: Info label updates with new size
✅ **Responsive**: Size changes apply to next ROI click
✅ **Professional UI**: Integrated seamlessly with mode selection

---

## Implementation Details

### Instance Variable
```python
class MainWindow:
    def __init__(self):
        # ...existing code...
        self.click_select_size = 40  # Default 40×40
```

### UI Control Creation
```python
# Label
size_label = QLabel("Size:")
size_label.setStyleSheet("font-size: 10px; font-weight: bold;")

# Input box (QSpinBox)
self.click_size_input = QSpinBox()
self.click_size_input.setMinimum(10)          # Min: 10 pixels
self.click_size_input.setMaximum(200)         # Max: 200 pixels
self.click_size_input.setValue(40)            # Default: 40
self.click_size_input.setStyleSheet("font-size: 10px; padding: 2px;")
self.click_size_input.setMinimumWidth(50)     # Compact width
self.click_size_input.valueChanged.connect(self.on_click_size_changed)

# Add to layout
mode_layout.addWidget(self.radio_drag)
mode_layout.addWidget(self.radio_click)
mode_layout.addWidget(size_label)
mode_layout.addWidget(self.click_size_input)
mode_layout.addStretch()
```

### Callback Method
```python
def on_click_size_changed(self):
    """Handle click select size change"""
    # Get new size from input box
    self.click_select_size = self.click_size_input.value()
    
    # Update radio button label dynamically
    self.radio_click.setText(f"Click ({self.click_select_size}×{self.click_select_size})")
    
    # Update status if click mode is active
    if self.radio_click.isChecked():
        self.info_label.setText(
            f"Selection Mode: Click {self.click_select_size}×{self.click_select_size} (single click to select area)"
        )
```

### MousePress Event Update
```python
def mousePressEvent(self, event):
    if current_mode == "click":
        # Get size from parent window
        size = 40  # Default
        if self.parent_window and hasattr(self.parent_window, 'click_select_size'):
            size = self.parent_window.click_select_size
        
        half_size = size // 2  # Half for centering
        
        # Calculate area centered at click
        center_x = int(click_pos.x() / self.zoom_level)
        center_y = int(click_pos.y() / self.zoom_level)
        
        x = max(0, center_x - half_size)
        y = max(0, center_y - half_size)
        w = size
        h = size
        
        # ...rest of calculation...
```

---

## User Workflow

### Setting Custom Size

**Step 1**: Select "Click" mode
```
⦾ Click (40×40)
```

**Step 2**: Change size in input box
```
[Size:] [60]  ← User types 60
```

**Step 3**: Observe updates
```
- Radio label changes: "Click (60×60)"
- Status shows: "Selection Mode: Click 60×60"
```

**Step 4**: Click on image
```
User clicks → Selects 60×60 area centered at click
```

---

## Dynamic Label Update

### Example Sequence

**Initial State:**
```
[⦾ Click (40×40)] [Size:] [40]
```

**User Changes to 50:**
```
Input box: 40 → 50
↓
Trigger: valueChanged signal
↓
Callback: on_click_size_changed()
↓
Updates:
- self.click_select_size = 50
- Radio label: "Click (50×50)"
- Status: "Selection Mode: Click 50×50 ..."
```

**User Changes to 80:**
```
Input box: 50 → 80
↓
Radio label: "Click (80×80)"
↓
Next click: Selects 80×80 area
```

---

## Visual Display

### Selection Mode Section
```
┌────────────────────────────────────────────┐
│ Selection Mode:                            │
│ [⦿ Drag Select] [⦾ Click (40×40)]         │
│                  [Size:] [40] [    ]      │
│                           ↑     ↑         │
│                         Input  Stretch    │
└────────────────────────────────────────────┘
```

### After User Changes Size to 50
```
┌────────────────────────────────────────────┐
│ Selection Mode:                            │
│ [⦿ Drag Select] [⦾ Click (50×50)]         │ ← Updated!
│                  [Size:] [50] [    ]      │ ← Updated!
└────────────────────────────────────────────┘
```

---

## Input Validation

### Range Constraints
| Constraint | Value | Reason |
|-----------|-------|--------|
| **Minimum** | 10 px | Practical minimum |
| **Maximum** | 200 px | Reasonable maximum |
| **Default** | 40 px | Standard size |
| **Step** | 1 px | Fine-grained control |

### Examples
```
Set to 15 px  → Selects 15×15 (smallest practical)
Set to 100 px → Selects 100×100 (larger area)
Set to 200 px → Selects 200×200 (maximum)
```

---

## Status Message Updates

### When Size Changes
```python
# Before: "Selection Mode: Click 40×40 ..."
# After user sets to 60: "Selection Mode: Click 60×60 ..."
```

### When Mode Changes
```python
# With size 50:
# Drag → Status: "Selection Mode: Drag Select (draw rectangle)"
# Click → Status: "Selection Mode: Click 50×50 (single click...)"
```

---

## Code Changes Summary

| Component | Change | Status |
|-----------|--------|--------|
| **Imports** | Added QSpinBox | ✅ |
| **MainWindow.__init__** | Added click_select_size variable | ✅ |
| **init_ui()** | Added size input components | ✅ |
| **on_click_size_changed()** | NEW callback method | ✅ |
| **on_selection_mode_changed()** | Updated to use new size | ✅ |
| **mousePressEvent()** | Updated to use dynamic size | ✅ |

---

## Benefits

✅ **Flexibility**: Users can choose any size from 10-200 pixels
✅ **Real-time Feedback**: Instant label and status updates
✅ **Professional UI**: Compact, well-organized interface
✅ **Dynamic**: Changes apply immediately to next selection
✅ **Validation**: Input bounds enforced automatically
✅ **User-Friendly**: Intuitive control

---

## Usage Examples

### Example 1: Small ROI (20×20)
```
1. Set Size input: 20
2. Radio changes to: "Click (20×20)"
3. Click on image
4. 20×20 area selected
```

### Example 2: Large ROI (100×100)
```
1. Set Size input: 100
2. Radio changes to: "Click (100×100)"
3. Click on image
4. 100×100 area selected
```

### Example 3: Custom Size (73×73)
```
1. Set Size input: 73
2. Radio changes to: "Click (73×73)"
3. Click on image
4. 73×73 area selected (odd number works!)
```

---

## Status

✅ **Implementation**: Complete
✅ **UI Control**: QSpinBox added
✅ **Validation**: Min/max constraints
✅ **Callback**: Properly connected
✅ **Dynamic Updates**: Working
✅ **Production Ready**: Yes

---

**File**: `/Users/samlai/Local_2/agent_test/SFR_app_v2.py` (1110+ lines)
**Date**: November 29, 2025
**Status**: ✅ Complete and verified

