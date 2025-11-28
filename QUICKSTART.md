## Quick Start Guide: Image Viewer with RAW Support

### Installation (First Time Only)

```bash
# Navigate to the project directory
cd /Users/samlai/Local_2/agent_test

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
python image_viewer_app.py
```

---

## Working with Different Image Formats

### Standard Images (PNG, JPEG, GIF, BMP, TIFF, WebP)

1. Click "Open Image"
2. Select your image file
3. Use Zoom In/Out or Fit buttons to adjust view
4. That's it!

### RAW Files

RAW files require configuration because they're just binary data.

#### Step 1: Open a RAW File
Click "Open Image" and select a `.raw` file

#### Step 2: Configure Parameters

A dialog will appear asking for:

**Width & Height**
- Enter the image dimensions in pixels
- Example: 512 × 512

**Data Type**
- **uint8**: 8-bit unsigned (0-255) - most common
- **uint16**: 16-bit unsigned (0-65535) - high-precision
- **float32**: 32-bit floating point - scientific data

**Quick Presets**
- Click predefined sizes (512×512, 1024×1024, 2048×2048)

#### Step 3: View & Adjust
Once loaded, use zoom and fit options like any other image

---

## Creating Test RAW Files

To get started with RAW images, generate sample files:

```bash
python create_sample_raw.py
```

This creates 5 sample RAW files with different patterns:
- Gradient (uint8)
- Checkerboard (uint8)
- Circle (uint8)
- Gradient (uint16)
- Sine Wave (float32)

Then open them in the Image Viewer!

---

## Troubleshooting

### Can't find the right dimensions?

Calculate based on file size:

```
Height = File Size / (Width × bytes per pixel)
```

**Example:**
- File: 262,144 bytes
- Data type: uint8 (1 byte per pixel)
- You know width is 512
- Height = 262,144 / 512 = 512

So it's a 512×512 image.

### File too small error?

The file doesn't have enough data for the dimensions you specified.

**Solution:**
- Double-check your width and height
- Verify the data type (uint16 is 2x larger than uint8!)

### Image looks like noise?

Probably wrong data type or byte order.

**Try:**
1. Different data types (uint8 → uint16 → float32)
2. Different dimensions
3. Check your RAW file format documentation

---

## Features

✅ Open and display standard image formats  
✅ Open and configure RAW binary files  
✅ Zoom in/out on any image  
✅ Fit images to window  
✅ Display image information  
✅ Simple, intuitive interface  

---

## Keyboard Navigation

- **Alt + F**: File menu
- **Alt + E**: Edit menu
- **Alt + H**: Help menu

---

## Need Help?

Check the full README.md for detailed documentation about RAW files and troubleshooting.

