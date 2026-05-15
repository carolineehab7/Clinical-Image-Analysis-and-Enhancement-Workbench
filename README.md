# Clinical Image Analysis and Enhancement Workbench

A comprehensive desktop application for medical image analysis and enhancement. This workbench provides researchers, clinicians, and engineers with powerful tools for processing, analyzing, and enhancing clinical images including DICOM files and standard image formats.

## Overview

The Clinical Image Analysis and Enhancement Workbench is a spatial and frequency domain image processing toolkit built with Python, offering a modern GUI and programmatic access to advanced image processing operations. It is designed to handle medical imaging workflows including noise synthesis and reduction, edge detection, histogram analysis, frequency-domain filtering, ROI statistics, and custom processing pipelines.

## Features

### Core Image Processing

- **Convolution & Filtering**: Apply custom kernels and standard filters (Average, Gaussian, Median)
- **Edge Detection**: Sobel edge detection with gradient magnitude, Gx, and Gy visualization
- **Histogram Analysis**: Compute, visualize, and equalize image histograms (global and CLAHE-style)
- **Noise Synthesis**: Add Gaussian, Salt & Pepper, Speckle, and Uniform noise with configurable parameters
- **Noise Reduction**: Smoothing filters (Average, Gaussian, Median) to restore noisy images
- **Frequency Domain**: 2D FFT spectrum display with Ideal and Butterworth notch filtering
- **ROI Analysis**: Click-and-drag region-of-interest selection with local histogram and statistics
- **Image Statistics**: Mean, variance, and pixel-count analysis per selected region
- **Normalization**: Intensity normalization to [0, 255]
- **Color Space Conversion**: RGB to Grayscale conversion
- **Interpolation**: Image resizing via various interpolation methods
- **Processing Pipelines**: Chain multiple operations together with full undo history
- **Morphological Operations** *(bonus)*: Erosion, Dilation, Opening, Closing, and Boundary extraction on binary images

### User Interface

- Modern GUI built with CustomTkinter
- **Smoothing Filter Panel**: Select filter type and kernel size with noise-type hints
- **Edge Detection Panel**: Run Sobel detection and view Gx, Gy, and magnitude in one window
- **Noise Panel**: Add four types of synthetic noise with adjustable parameters
- **FFT & Notch Filter Panel**: Inspect the frequency spectrum and interactively remove periodic noise
- **ROI Panel**: Draw a region of interest and open a local statistics popup
- **Histogram Window**: Full-image histogram visualization
- **Statistics Window**: Bar-chart histogram + mean and variance for any ROI
- **Pipeline Panel**: View and manage the multi-step processing history
- **Zoom Panel**: Magnify and inspect image details
- **Spectrum Window**: Interactive FFT display with clickable notch placement
- **Theming**: Professional dark theme throughout

### File Format Support

- DICOM files (`.dcm`) for medical imaging via pydicom
- Standard image formats (PNG, JPG, BMP, etc.) via Pillow

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 2 GB (4 GB+ recommended for large medical images)
- **Display**: 1024×768 minimum resolution

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/carolineehab7/Clinical-Image-Analysis-and-Enhancement-Workbench.git
cd Clinical-Image-Analysis-and-Enhancement-Workbench
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dependencies

- **customtkinter** (≥5.2.0): Modern Python UI toolkit for GUI
- **pydicom** (≥2.4.0): Read and work with DICOM medical images
- **Pillow** (≥10.0.0): Python Imaging Library for image I/O
- **numpy** (≥1.24.0): Numerical computing library

## Quick Start

```bash
python main.py
```

The GUI window will open, allowing you to:

1. Load an image file (DICOM or standard formats)
2. Apply smoothing filters and preview results
3. Detect edges and view gradient components
4. Add or remove noise
5. Inspect the FFT spectrum and apply notch filters
6. Draw an ROI and analyze local statistics
7. Build multi-step processing pipelines
8. Export processed images

## Project Structure

```
.
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── core/                            # Core image processing modules
│   ├── pipeline.py                  # Processing pipeline and undo history
│   │
│   ├── filters/                     # Spatial-domain filtering
│   │   ├── convolution.py           # Base convolution engine
│   │   ├── smoothing.py             # Average, Gaussian, and Median filters
│   │   ├── median.py                # Optimised median filter (stride tricks)
│   │   └── edge_detection.py        # Sobel edge detection
│   │
│   ├── transforms/                  # Image transformations
│   │   ├── noise.py                 # Gaussian, Salt & Pepper, Speckle, Uniform noise
│   │   └── interpolation.py         # Image resizing and interpolation
│   │
│   ├── frequency/                   # Frequency-domain processing
│   │   ├── fft.py                   # 2D FFT and log-magnitude spectrum
│   │   └── notch.py                 # Ideal and Butterworth notch-reject masks
│   │
│   ├── enhancement/                 # Image enhancement utilities
│   │   ├── histogram.py             # Histogram computation and equalization
│   │   ├── normalization.py         # Intensity normalization
│   │   └── color_conversion.py      # RGB-to-grayscale conversion
│   │
│   └── analysis/                    # Quantitative image analysis
│       ├── roi.py                   # ROI extraction
│       └── statistics.py            # Mean, variance, local histogram
│
├── gui/                             # User interface components
│   ├── main_window.py               # Main application window
│   ├── theme.py                     # Colours, fonts, and styling constants
│   │
│   ├── panels/                      # Left-sidebar control panels
│   │   ├── filter_panel.py          # Smoothing filter controls
│   │   ├── EdgeDetection_panel.py   # Edge detection controls
│   │   ├── noise_panel.py           # Noise synthesis controls
│   │   ├── fft_panel.py             # FFT & notch filter launcher
│   │   ├── roi_panel.py             # ROI draw / analyze controls
│   │   ├── pipeline_panel.py        # Pipeline history viewer
│   │   └── zoom_panel.py            # Image zoom controls
│   │
│   └── windows/                     # Pop-up analysis windows
│       ├── histogram_window.py      # Full-image histogram display
│       ├── statistics_window.py     # ROI histogram + mean / variance
│       └── spectrum_window.py       # Interactive FFT spectrum display
│
└── bonus/                           # Bonus morphological processing
    ├── morphology.py                # Erosion, Dilation, Opening, Closing, Boundary
    └── morphology_panel.py          # GUI panel for morphological operations
```

## Core Modules

### `core/filters/`

- **convolution.py** — Base 2D convolution engine used by all kernel-based filters.
- **smoothing.py** — Average (box), Gaussian, and Median filter implementations built from scratch.
- **median.py** — Optimised median filter using NumPy stride tricks for per-channel processing.
- **edge_detection.py** — Sobel operator computing Gx, Gy, and gradient magnitude.

### `core/transforms/`

- **noise.py** — Adds Gaussian, Salt & Pepper, Speckle, and Uniform noise with configurable parameters and optional random seeds.
- **interpolation.py** — Image resizing via various interpolation methods.

### `core/frequency/`

- **fft.py** — Computes a centered 2D FFT and generates a log-magnitude display image.
- **notch.py** — Builds Ideal and Butterworth notch-reject masks for periodic noise removal.

### `core/enhancement/`

- **histogram.py** — Histogram computation, global equalization, and block-based CLAHE-style equalization.
- **normalization.py** — Normalizes pixel intensities to [0, 255].
- **color_conversion.py** — Converts RGB images to grayscale.

### `core/analysis/`

- **roi.py** — Extracts a rectangular region of interest from an image.
- **statistics.py** — Computes local histogram, mean, and variance for any image region.

### `bonus/morphology.py`

Binary morphological operations (Erosion, Dilation, Opening, Closing, Boundary extraction) with configurable square or cross structuring elements.

## Usage Examples

### Programmatic Image Processing

```python
from core.io.image_io import load_image
from core.filters.edge_detection import sobel_filter
from core.filters.smoothing import gaussian_smoothing_filter
from core.transforms.noise import add_salt_pepper_noise
from core.frequency.fft import compute_fft

# Load image
image = load_image("path/to/medical/image.dcm")

# Add salt-and-pepper noise, then smooth
noisy = add_salt_pepper_noise(image, amount=0.05)
smoothed = gaussian_smoothing_filter(noisy, kernel_size=5, variance=1.0)

# Sobel edge detection
edges, gx, gy = sobel_filter(image)

# Frequency-domain inspection
F_shifted, mag_display = compute_fft(image)
```

### Using the GUI

1. Launch: `python main.py`
2. **Load Image** — use the File menu to open a DICOM or standard image
3. **Apply Filters** — select Average, Gaussian, or Median in the Filter Panel
4. **Add Noise** — choose noise type and parameters in the Noise Panel
5. **Detect Edges** — click Run in the Edge Detection Panel to view Gx, Gy, and magnitude
6. **Frequency Domain** — open the Spectrum Window, click notch points, and apply the mask
7. **ROI Analysis** — draw a region, then click Analyze to view local histogram and statistics
8. **Pipeline** — review all applied operations in the Pipeline Panel
9. **Export** — save the processed image in your desired format
