# Image Captioning System with Object Detection

**Minor Project (3rd Semester)**
**Author:** Harsh Mishra

## Overview
3 

To improve accuracy and context, the system integrates:
*   **VGG16**: For extracting visual features from images.
*   **YOLOv8**: For detecting specific objects (e.g., "person", "dog") to provide semantic context.
*   **Beam Search**: For generating more grammatically correct and coherent sentences compared to standard greedy search.

## Features
*   **Dual-Input Model**: Processes both image visual features and detected object tags.
*   **Object Detection**: Uses YOLOv8 to identify 80+ object classes.
*   **Beam Search Inference**: Toggle between fast (Greedy) and accurate (Beam Search) caption generation.
*   **Modern GUI**: Built with `customtkinter` for a sleek, dark-mode user interface.

## Installation

1.  **Clone the repository** (or download the source code).
2.  **Install dependencies**:
    ```bash
    pip install tensorflow pandas matplotlib scikit-learn tqdm pillow ultralytics customtkinter
    ```
3.  **Prepare Data**:
    *   Place Flickr8k images in `data/Images/`.
    *   Place `captions.txt` in `data/`.

## Usage

### 1. Running the GUI (Recommended)
The easiest way to use the project is via the Graphical User Interface.
```bash
python gui/app.py
```
*   **Upload Image**: Select an image from your computer.
*   **Generate Caption**: Click the button to see the result.
*   **Beam Search**: Toggle the switch to enable smarter caption generation.

### 2. Training the Model (Optional)
If you want to retrain the model from scratch:
```bash
python src/train.py
```
*   This will process images, extract features/objects, and train the model for 20 epochs.
*   The trained model will be saved to `models/model_final.keras`.

### 3. Object Extraction
To re-run object detection on all images:
```bash
python src/object_extractor.py
```

## Project Structure
*   `data/`: Images and captions.
*   `models/`: Saved models (`.keras`) and pickle files (`tokenizer.pkl`, `features.pkl`, `objects.pkl`).
*   `src/`: Source code for training, data loading, and inference.
*   `gui/`: Application code (`app.py`).
*   `docs/`: Project documentation and reports.

## Credits
Developed by **Harsh Mishra**.
Based on the Flickr8k dataset.
