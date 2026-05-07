# Clinical Image Analysis and Enhancement Workbench

A comprehensive desktop application for medical image analysis and enhancement. This workbench provides researchers, clinicians, and engineers with powerful tools for processing, analyzing, and enhancing clinical images including DICOM files and standard image formats.

## Overview

The Clinical Image Analysis and Enhancement Workbench is a spatial domain image processing toolkit built with Python, offering both a user-friendly GUI and programmatic access to advanced image processing operations. It's designed to handle medical imaging workflows including noise reduction, edge detection, histogram analysis, filtering, and custom processing pipelines.

## Features

### Core Image Processing

- **Convolution & Filtering**: Apply custom kernels and standard filters
- **Edge Detection**: Sobel edge detection and multiple edge-finding algorithms
- **Histogram Analysis**: Compute, visualize, and analyze image histograms
- **Noise Reduction**: Multiple noise reduction filters including median filtering
- **Image Enhancement**: Normalization and intensity adjustment
- **Interpolation**: Various interpolation methods for image resizing
- **Color Space Conversion**: RGB to Grayscale conversion
- **Processing Pipelines**: Chain multiple operations together

### User Interface

- Modern GUI built with CustomTkinter for an intuitive experience
- **Image Panels**: Load and display clinical images
- **Edge Detection Panel**: Interactive edge detection with parameter adjustment
- **Filter Panel**: Apply and preview various filters in real-time
- **Histogram Window**: Analyze and display image histograms
- **Noise Reduction Panel**: Configure and apply noise reduction
- **Pipeline Panel**: Create and execute multi-step processing workflows
- **Zoom Controls**: Magnify and inspect image details
- **Theming**: Professional dark/light theme support

### File Format Support

- DICOM files (.dcm) for medical imaging
- Standard image formats (PNG, JPG, BMP, etc.) via Pillow

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 2GB (4GB+ recommended for large medical images)
- **Display**: 1024x768 minimum resolution

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Clinical-Image-Analysis-and-Enhancement-Workbench.git
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
- **Pillow** (≥10.0.0): Python Imaging Library for image processing
- **numpy** (≥1.24.0): Numerical computing library

## Quick Start

### Running the Application

```bash
python main.py
```

The GUI window will open, allowing you to:

1. Load an image file (DICOM or standard formats)
2. Apply various filters and processing operations
3. Analyze histograms and image statistics
4. Create custom processing pipelines
5. Export processed images

## Project Structure

```
.
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── core/                      # Core image processing modules
│   ├── convolution.py         # Convolution operations
│   ├── filters.py             # Standard image filters
│   ├── histogram.py           # Histogram computation and analysis
│   ├── image_io.py            # Image loading and saving
│   ├── interpolation.py       # Image resizing and interpolation
│   ├── Median.py              # Median filtering
│   ├── noise.py               # Noise reduction filters
│   ├── normalization.py       # Image normalization
│   ├── pipeline.py            # Processing pipeline framework
│   ├── RGB_to_gray.py         # Color space conversion
│   └── Sobel.py               # Sobel edge detection
│
└── gui/                       # User Interface components
    ├── main_window.py         # Main application window
    ├── EdgeDetection_panel.py  # Edge detection interface
    ├── filter_panel.py         # Filter application panel
    ├── histogram_window.py     # Histogram visualization
    ├── noise_panel.py          # Noise reduction panel
    ├── pipeline_panel.py       # Pipeline creation panel
    ├── zoom_panel.py           # Image zoom controls
    └── theme.py                # UI theming and styling
```

## Core Modules

### `core/image_io.py`

Handles loading and saving images in various formats, including DICOM support.

### `core/histogram.py`

Computes histogram data and provides statistical analysis of image intensity distributions.

### `core/convolution.py`

Implements convolution operations for kernel-based filtering.

### `core/filters.py`

Standard image filters (Gaussian, median, morphological operations, etc.).

### `core/Sobel.py`

Sobel edge detection for identifying image edges and boundaries.

### `core/noise.py`

Noise reduction algorithms for improving image quality.

### `core/normalization.py`

Intensity normalization and enhancement techniques.

### `core/interpolation.py`

Image resizing and interpolation methods.

## Usage Examples

### Loading and Processing an Image Programmatically

```python
from core.image_io import load_image
from core.Sobel import sobel_edge_detection
from core.filters import apply_gaussian_filter

# Load image
image = load_image("path/to/medical/image.dcm")

# Apply Sobel edge detection
edges = sobel_edge_detection(image)

# Apply Gaussian filter for smoothing
smoothed = apply_gaussian_filter(image, kernel_size=5)
```

### Using the GUI

1. Launch the application: `python main.py`
2. **Load Image**: Use the File menu or drag-and-drop to load an image
3. **Apply Filters**: Use the Filter Panel to select and apply filters
4. **Detect Edges**: Use the Edge Detection Panel with adjustable parameters
5. **Analyze**: View histograms and image statistics in the Histogram Window
6. **Create Pipeline**: Chain operations in the Pipeline Panel
7. **Export**: Save processed images in your desired format
