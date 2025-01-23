# Satellite Port Detector

## Overview
Satellite Port Detector is a Python application with a Tkinter-based graphical user interface (GUI) for analyzing satellite imagery. The application provides several image processing and detection features to help analyze satellite images of ports.

## Features
- Load satellite images from various file formats
- Detect rotation angles of circular objects
- Find circles within image crops
- Apply perspective transformations
- Simulate camera movement and perspective changes

## Prerequisites
- Python 3.7+
- OpenCV
- NumPy
- Pillow (PIL)
- Tkinter (usually comes with Python standard library)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/satellite-port-detector.git
cd satellite-port-detector
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python tkcvchv4.py
```

### Workflow
1. **Load Image**: Click "Load Image" to select a satellite image
2. **Detect Rotation**: Automatically detect the rotation angle of circular objects
3. **Find Circles**: Scan image crops to detect circular features
4. **Apply Perspective Transform**: Adjust image perspective
5. **Simulate Camera Movement**: Animate perspective changes

## Dependencies
See `requirements.txt` for a complete list of Python package dependencies.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Insert your license here, e.g., MIT License]

## Disclaimer
This tool is for research and educational purposes. Ensure compliance with local regulations when using satellite imagery.
