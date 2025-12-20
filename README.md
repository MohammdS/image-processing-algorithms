A modular collection of computer vision and image processing algorithms implemented in Python. This repository serves as a workbench for exploring fundamental concepts in image processing 

## 📂 Current Modules

### `histogram_modules.py`
This is the core tool currently available. It provides an interactive command-line interface (CLI) to perform the following tasks:

* **Histogram Equalization:**
    * **Global Equalization:** Enhances overall contrast using standard CDF mapping.
    * **Adaptive Histogram Equalization (AHE):** Enhances local contrast using tile-based processing and bilinear interpolation.
* **Template Matching:**
    * **Histogram-Based (EMD):** specific object detection using Earth Mover's Distance on 1D histograms.
    * **Matching by Tone Mapping (MTM):** A robust template matching algorithm (using Slice Transform) that handles non-linear brightness changes.
* **Robustness Testing:** Automatically generates image variants (brightness/contrast shifts) to test how well the matching algorithms perform under stress.

## 📁 Data Management

The tool includes a smart file selection system designed for ease of use:

1.  **The `Data` Folder:**
    If you create a folder named `Data` in the same directory as the script and place your images there, the program will automatically detect them. When you run the tool, it will present a numbered list of available images for quick selection.

2.  **Manual Input:**
    If you do not have a `Data` folder, or if you want to use an image stored elsewhere on your machine, the tool allows you to manually paste the full file path.

## 🚀 How to Run

### 1. Prerequisites
Make sure you have Python installed along with the necessary libraries:

```bash
pip install numpy opencv-python matplotlib
