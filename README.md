# BackgroundImageRemover - Image Converter, Optimizer & Background Remover

A modern, stylish desktop application for batch image conversion, resizing, optimization, and intelligent background removal, built with PyQt5, Pillow, and OpenCV. **BackgroundImageRemover** is designed to convert and optimize images (PNG and WEBP) for web use, making your images lighter and faster to load online, while also providing professional-grade background removal capabilities.

## Features

- **Batch Image Conversion:** Convert images in bulk between PNG and WEBP formats
- **Size Optimization:** Resize images to target dimensions (100px to 700px) while maintaining quality
- **Smart Background Removal:** Production-ready white/light background removal using multi-strategy detection (LAB color space analysis, adaptive Canny edge detection, HSV saturation detection) with GrabCut-inspired edge refinement
- **Image Renaming:** Rename images based on their subfolder names for better organization
- **Progress Bar:** Visual feedback during long-running operations
- **Custom UI Elements:** Includes blinking buttons, custom alerts, and a frameless, translucent window
- **Modern Interface:** Clean, modern UI with custom styling and animations
- **Cross-Platform:** Designed for Windows, but can be adapted for other platforms

## Screenshots

> _Add screenshots of your app here (UI, before/after conversion, background removal results, etc.)_

## Getting Started

### Prerequisites

- Python 3.12+
- [pip](https://pip.pypa.io/en/stable/installation/)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/BackgroundImageRemover.git
   cd BackgroundImageRemover
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up virtual environment (recommended):**

   ```bash
   python -m venv appenv
   appenv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

### Running the App

```bash
python main.py
```

The main window will appear, allowing you to select an image folder, choose output format and size, and start the conversion process. The app now includes smart background removal capabilities for creating transparent PNG images.

### Packaging as an EXE (Windows)

This project includes build scripts for easy packaging. To build a standalone executable:

**Using batch file:**
```bash
build.bat
```

**Using PowerShell:**
```powershell
.\build.ps1
```

**Manual build:**
```bash
pyinstaller build_exe.spec
```

The output will be in the `dist/` directory as `BackgroundImageRemover.exe`.

## Project Structure

```
BackgroundImageRemover/
│
├── main.py                # Main application entry point (PyQt5 GUI)
├── requirements.txt       # Python dependencies
├── build_exe.spec         # PyInstaller spec for Windows packaging
├── build.bat              # Windows batch build script
├── build.ps1              # PowerShell build script
├── main.spec              # Alternative PyInstaller spec
├── src/
│   ├── assets/            # Image conversion, renaming, and utility scripts
│   │   ├── convertor.py   # Main image conversion logic
│   │   ├── smart_background_remover.py # Intelligent background removal
│   │   ├── update_file.py # File update utilities
│   │   ├── transform_string.py # String transformation utilities
│   │   ├── reorder.py     # File reordering utilities
│   │   ├── rename_files.py # File renaming utilities
│   │   ├── count_files_in_directory.py # File counting utilities
│   │   └── is_image.py    # Image validation utilities
│   ├── uiitems/           # Custom UI widgets
│   │   ├── close_button.py # Custom close button
│   │   ├── blink_button.py # Animated blinking button
│   │   ├── text_box.py    # Custom text input
│   │   ├── preview_box.py # Image preview component
│   │   ├── notification_bar.py # Notification display
│   │   ├── file_input.py  # File input component
│   │   ├── dash_line.py   # Decorative dash line
│   │   ├── custom_alert.py # Custom alert dialogs
│   │   └── collapsible_box.py # Collapsible UI sections
│   └── widgets/           # Main application widgets
│       ├── drag_drop.py   # Drag and drop functionality
│       ├── img_resizer.py # Image resizing widget
│       ├── img_renamer.py # Image renaming widget
│       ├── initiation_files_input.py # File input widget
│       ├── select_initiation_csv.py # CSV selection widget
│       └── login.py       # Login widget
├── static/
│   ├── logo_imgs/         # App icons and logos
│   │   ├── cover.png      # App cover image
│   │   └── favicon.ico    # App icon
│   └── styles.css         # CSS styling (for documentation)
├── appenv/                # Virtual environment directory
└── README.md
```

## Dependencies

- **PyQt5** - GUI framework
- **Pillow (PIL)** - Image processing
- **OpenCV (cv2)** - Advanced image processing and background removal
- **NumPy** - Numerical operations for image processing
- **Matplotlib** - Visualization and plotting
- **python-dotenv** - Environment variable management
- **pyinstaller** - For creating standalone executables

## Background Removal Features

The smart background remover is a production-ready solution that includes:

- **Multi-Strategy Detection System:** 
  - LAB color space analysis for precise white/light background detection
  - Adaptive Canny edge detection with dynamic thresholds based on image statistics
  - HSV saturation-based detection for colored objects
  - Intelligent combination of all strategies for optimal results

- **Advanced Edge Refinement:** 
  - GrabCut-inspired trimap generation for uncertain pixel classification
  - Color similarity analysis in LAB color space for edge refinement
  - Morphological operations for noise reduction and hole filling
  - Median blur smoothing for clean final edges

- **Intelligent Object Preservation:** 
  - Automatic object size filtering (minimum 0.1% of image area)
  - Connected component analysis to remove noise
  - Adaptive thresholding based on image brightness statistics
  - Edge pixel analysis for background detection

- **Quality Features:**
  - Bilateral filtering for noise reduction while preserving edges
  - Dynamic threshold calculation based on image characteristics
  - Automatic hole filling within detected objects
  - High-quality RGBA PNG output with transparency

- **Batch Processing:** Process multiple images with transparent backgrounds
- **Production Ready:** Comprehensive error handling and logging
- **Visualization Support:** Built-in result visualization with matplotlib

## Build Scripts

The project includes several build scripts for convenience:

- **build.bat** - Windows batch script for building executable
- **build.ps1** - PowerShell script for building executable
- **build_exe.spec** - Main PyInstaller specification file
- **main.spec** - Alternative PyInstaller specification file

## Customization

- **UI Styling:** Modify the stylesheet in `main.py` for custom colors and layout
- **Image Formats:** Extend `src/assets/convertor.py` to support more formats if needed
- **Background Removal:** Customize detection parameters in `src/assets/smart_background_remover.py`
- **UI Components:** Customize widgets in `src/uiitems/` and `src/widgets/` directories

## License

MIT License. See [LICENSE](LICENSE) for details.
