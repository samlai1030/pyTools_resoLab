"""
Image File Viewer Application
A simple GUI application to open and view various image formats
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np


class ImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        self.current_image = None
        self.photo_image = None
        self.image_path = None

        # Create menu bar
        self.create_menu_bar()

        # Create main frame
        self.create_ui()

    def create_menu_bar(self):
        """Create the menu bar with File options"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Zoom In", command=self.zoom_in)
        edit_menu.add_command(label="Zoom Out", command=self.zoom_out)
        edit_menu.add_command(label="Fit to Window", command=self.fit_to_window)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_ui(self):
        """Create the main UI elements"""
        # Top frame for buttons
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(top_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Fit", command=self.fit_to_window).pack(side=tk.LEFT, padx=5)

        # Info label
        self.info_label = tk.Label(top_frame, text="No image loaded", fg="gray")
        self.info_label.pack(side=tk.LEFT, padx=20)

        # Canvas for image display
        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom frame for status
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.status_label = tk.Label(bottom_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X)

    def open_image(self):
        """Open an image file dialog and load the image"""
        filetypes = (
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.webp *.raw"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("GIF files", "*.gif"),
            ("BMP files", "*.bmp"),
            ("TIFF files", "*.tiff"),
            ("WebP files", "*.webp"),
            ("RAW files", "*.raw"),
            ("All files", "*.*"),
        )

        filepath = filedialog.askopenfilename(
            title="Open Image",
            filetypes=filetypes
        )

        if filepath:
            self.load_image(filepath)

    def load_image(self, filepath):
        """Load and display an image"""
        try:
            self.image_path = filepath

            # Check if it's a RAW file
            if filepath.lower().endswith('.raw'):
                self.load_raw_image(filepath)
            else:
                self.current_image = Image.open(filepath)
                self.display_image(self.current_image)

                # Update info
                filename = os.path.basename(filepath)
                size = self.current_image.size
                self.info_label.config(text=f"{filename} ({size[0]}x{size[1]})")
                self.status_label.config(text=f"Loaded: {filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{str(e)}")
            self.status_label.config(text="Error loading image")

    def load_raw_image(self, filepath):
        """Load and display a RAW image file"""
        try:
            # Get file size
            file_size = os.path.getsize(filepath)

            # Create dialog for RAW parameters
            dialog = RawImageDialog(self.root, file_size)
            if dialog.result is None:
                return

            width, height, data_type = dialog.result

            # Read RAW file
            with open(filepath, 'rb') as f:
                raw_data = f.read()

            # Parse based on data type
            if data_type == "uint8":
                expected_size = width * height
                if len(raw_data) >= expected_size:
                    data = np.frombuffer(raw_data[:expected_size], dtype=np.uint8)
                else:
                    messagebox.showerror("Error", f"File too small. Expected {expected_size} bytes, got {len(raw_data)}")
                    return

            elif data_type == "uint16":
                expected_size = width * height * 2
                if len(raw_data) >= expected_size:
                    data = np.frombuffer(raw_data[:expected_size], dtype=np.uint16)
                    # Convert to 8-bit for display
                    data = (data / 256).astype(np.uint8)
                else:
                    messagebox.showerror("Error", f"File too small. Expected {expected_size} bytes, got {len(raw_data)}")
                    return

            elif data_type == "float32":
                expected_size = width * height * 4
                if len(raw_data) >= expected_size:
                    data = np.frombuffer(raw_data[:expected_size], dtype=np.float32)
                    # Normalize to 0-255
                    data_min = data.min()
                    data_max = data.max()
                    if data_max > data_min:
                        data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                    else:
                        data = data.astype(np.uint8)
                else:
                    messagebox.showerror("Error", f"File too small. Expected {expected_size} bytes, got {len(raw_data)}")
                    return

            # Reshape to 2D array
            image_array = data.reshape((height, width))

            # Convert to PIL Image
            self.current_image = Image.fromarray(image_array, mode='L')
            self.display_image(self.current_image)

            # Update info
            filename = os.path.basename(filepath)
            self.info_label.config(text=f"{filename} ({width}x{height}) [{data_type}]")
            self.status_label.config(text=f"Loaded RAW: {filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load RAW image:\n{str(e)}")
            self.status_label.config(text="Error loading RAW image")

    def display_image(self, image):
        """Display image on canvas"""
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 500

        # Calculate scaling to fit canvas
        img_width, img_height = image.size
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(resized_image)

        # Clear canvas and add image
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.photo_image
        )

    def zoom_in(self):
        """Zoom in on the current image"""
        if self.current_image:
            width, height = self.current_image.size
            self.current_image = self.current_image.resize(
                (int(width * 1.2), int(height * 1.2)),
                Image.Resampling.LANCZOS
            )
            self.display_image(self.current_image)
            self.status_label.config(text="Zoomed in")

    def zoom_out(self):
        """Zoom out on the current image"""
        if self.current_image:
            width, height = self.current_image.size
            self.current_image = self.current_image.resize(
                (int(width / 1.2), int(height / 1.2)),
                Image.Resampling.LANCZOS
            )
            self.display_image(self.current_image)
            self.status_label.config(text="Zoomed out")

    def fit_to_window(self):
        """Fit image to window"""
        if self.image_path:
            self.load_image(self.image_path)

    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Image Viewer Application\n\n"
            "A simple image viewer for various image formats.\n\n"
            "Supported formats: PNG, JPEG, GIF, BMP, TIFF, WebP, RAW"
        )


class RawImageDialog:
    """Dialog for configuring RAW image parameters"""
    def __init__(self, parent, file_size):
        self.result = None
        self.file_size = file_size

        # Create top-level window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("RAW Image Parameters")
        self.dialog.geometry("400x300")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Info label
        info_text = f"File size: {file_size} bytes\n\nConfigure RAW image parameters:"
        tk.Label(self.dialog, text=info_text, justify=tk.LEFT).pack(padx=10, pady=10)

        # Width
        width_frame = tk.Frame(self.dialog)
        width_frame.pack(padx=10, pady=5, fill=tk.X)
        tk.Label(width_frame, text="Width (pixels):", width=15, anchor="w").pack(side=tk.LEFT)
        self.width_var = tk.StringVar(value="512")
        tk.Entry(width_frame, textvariable=self.width_var, width=20).pack(side=tk.LEFT, padx=5)

        # Height
        height_frame = tk.Frame(self.dialog)
        height_frame.pack(padx=10, pady=5, fill=tk.X)
        tk.Label(height_frame, text="Height (pixels):", width=15, anchor="w").pack(side=tk.LEFT)
        self.height_var = tk.StringVar(value="512")
        tk.Entry(height_frame, textvariable=self.height_var, width=20).pack(side=tk.LEFT, padx=5)

        # Data type
        dtype_frame = tk.Frame(self.dialog)
        dtype_frame.pack(padx=10, pady=5, fill=tk.X)
        tk.Label(dtype_frame, text="Data Type:", width=15, anchor="w").pack(side=tk.LEFT)
        self.dtype_var = tk.StringVar(value="uint8")
        dtype_menu = tk.OptionMenu(
            dtype_frame, self.dtype_var,
            "uint8", "uint16", "float32"
        )
        dtype_menu.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Info text about common sizes
        info_frame = tk.Frame(self.dialog)
        info_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        tk.Label(info_frame, text="Quick presets:", font=("Arial", 10, "bold")).pack(anchor="w")

        preset_frame = tk.Frame(info_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        tk.Button(preset_frame, text="512x512", command=lambda: self.set_size(512, 512)).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_frame, text="1024x1024", command=lambda: self.set_size(1024, 1024)).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_frame, text="2048x2048", command=lambda: self.set_size(2048, 2048)).pack(side=tk.LEFT, padx=2)

        # Buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(padx=10, pady=10, fill=tk.X)
        tk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)

        # Wait for dialog
        self.dialog.wait_window()

    def set_size(self, width, height):
        """Set width and height from preset"""
        self.width_var.set(str(width))
        self.height_var.set(str(height))

    def ok_clicked(self):
        """Handle OK button click"""
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            data_type = self.dtype_var.get()

            if width <= 0 or height <= 0:
                messagebox.showerror("Error", "Width and height must be positive")
                return

            self.result = (width, height, data_type)
            self.dialog.destroy()
        except ValueError:
            messagebox.showerror("Error", "Width and height must be valid integers")


def main():
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

