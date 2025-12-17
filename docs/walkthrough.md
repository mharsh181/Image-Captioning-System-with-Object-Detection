# Image Captioning Model Walkthrough

I have successfully built an image captioning model from scratch using the Flickr8k dataset. The pipeline includes data loading, feature extraction (VGG16), tokenization, model training (CNN-LSTM), and inference.

## 1. Data Preparation
- **Script**: `data_loader.py`
- **Function**: Loads `captions.txt`, cleans the text (lowercase, remove punctuation), and visualizes sample images.
- **Output**: `sample_0.png`, etc.

## 2. Feature Extraction
- **Script**: `feature_extractor.py`
- **Function**: Uses a pre-trained VGG16 model to extract features from images.
- **Note**: Currently configured to process a subset of 100 images for speed. You can remove the `limit` argument in the script to process the full dataset.
- **Output**: `features.pkl`

## 3. Tokenization
- **Script**: `tokenizer.py`
- **Function**: Fits a Keras Tokenizer on the captions to convert words to integers.
- **Output**: `tokenizer.pkl`

## 4. Model Training
- **Script**: `train.py`
- **Function**: Defines the CNN-LSTM model and trains it using a custom training loop.
- **Configuration**:
    - **Encoder**: VGG16 features (4096 dim) -> Dense (256 dim)
    - **Decoder**: Embedding -> LSTM (256 dim)
    - **Training**: Currently trains on the subset of 80 images for 5 epochs.
- **Output**: `model_final.keras`

## 5. Inference
- **Script**: `inference.py`
- **Function**: Generates a caption for a random image from the `Images` directory using the trained model.
- **Usage**: Run `python inference.py`.

## Next Steps
To get high-quality captions:
1.  **Process Full Dataset**: Modify `feature_extractor.py` and `train.py` to remove the `LIMIT = 100` restriction.
2.  **Train Longer**: Increase `epochs` in `train.py` (e.g., to 20 or 50).
3.  **Use GPU**: Ensure you have a GPU available for faster training on the full dataset.

## Verification Results
- **Training**: Successfully trained on full dataset (8091 images) for 50 epochs.
- **Evaluation**:
    - **BLEU-1**: 0.599
    - **BLEU-2**: 0.408
    - **BLEU-3**: 0.328
    - **BLEU-4**: 0.210
- **Inference**: Successfully generated captions. Run `python inference.py` to test.

## GUI Application
I have created a modern GUI application for the image captioning system.
- **Location**: `GUI_App/`
- **Run**: `python GUI_App/app.py`
- **Features**:
    - Upload any image.
    - View the image and generated caption.
    - Footer: "MADE BY HARSH MISHRA"

## Phase 4: Object Detection Integration
We have upgraded the model to use **YOLOv8** for object detection.

### Changes
*   **Dual Input Model**: The model now accepts Image Features (VGG16) AND Object Features (YOLO).
*   **Object Extraction**: `object_extractor.py` detects objects in all images.
*   **GUI Update**: The GUI now displays detected objects (e.g., "Detected: person, dog") and uses them to generate better captions.

### How to Run
1.  Run the GUI:
    ```bash
    python gui/app.py
    ```
2.  Upload an image.
3.  See the detected objects and the generated caption!

## Phase 5: Beam Search
We have added a smarter way to generate captions.

### How to Use
1.  Run the GUI as usual.
2.  You will see a new switch: **"Use Beam Search"**.
3.  **OFF (Default)**: Uses "Greedy Search". Fast, but might make grammar mistakes.
4.  **ON**: Uses "Beam Search". Slower, but explores multiple sentence possibilities to find the best one.
5.  Try generating captions with both to see the difference!
