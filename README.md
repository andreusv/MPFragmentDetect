# MPFragmentDetect

This repository provides a technical demonstration for the detection and segmentation of microplastic fragments using deep learning. The project evaluates and implements three distinct computer vision architectures tailored for microplastic analysis in high-resolution imagery.

## Overview of Models
The system supports three different tasks within the computer vision domain:

1. **YOLO11s (Object Detection)**: Optimised for rapid identification of fragments using bounding boxes.
2. **Mask R-CNN (Instance Segmentation)**: Provides individual masks for each detected fragment, allowing for precise morphological analysis.
3. **U-Net with EfficientNet-B3 (Semantic Segmentation)**: Implements a sliding-window inference strategy (stitching) to process high-resolution images without losing pixel-level detail.

## Installation

### Option A: Using Miniconda (Recommended)
To replicate the environment with all necessary dependencies, run the following commands:

```bash
conda env create -f environment.yml
conda activate mfr_detect
```

### Option B: Using Pip
Ensure you have Python 3.10+ installed, then run:

```bash
pip install -r requirements.txt
```

## Usage
The main.py script serves as the primary entry point. It automatically processes all images in the examples/ directory and saves results in the predictions/ folder.

Running YOLO11s
```Bash
python main.py --model yolo --weights models/yolo11s.pt
```
Running Mask R-CNN
```Bash
python main.py --model maskrcnn --weights models/maskrcnn.pth
```
Running U-Net
```Bash
python main.py --model unet --weights models/unet.pt
```

## Project Structure
- models/: Directory containing trained weights (.pt and .pth files).
- src/: Core logic including inference engines and visualization utilities.
- examples/: Sample images for technical demonstration.
- predictions/: Output directory where results are organized by model type.
- requirements.txt: List of Python dependencies.
- environment.yml: Conda environment configuration.
