# Image Captioning Model Implementation Plan

## Goal Description
Build an image captioning model from scratch using the Flickr8k dataset. The model will take an image as input and generate a textual description (caption). We will use a CNN-RNN (Encoder-Decoder) architecture.

**[NEW] Phase 4 Upgrade**: Integrate **Object Detection** to provide semantic context (detected objects) to the model, improving caption accuracy.

## User Review Required
- **Deep Learning Framework**: TensorFlow/Keras.
- **Object Detection**: I plan to use **YOLOv8** (via `ultralytics`) to detect 80 common object classes (Person, Dog, Cat, Car, etc.).
- **Architecture Change**: The model will now have **two inputs**:
    1.  Image Features (VGG16)
    2.  Object Tags (YOLO)

## Proposed Changes

### Data Processing
#### [MODIFY] [data_loader.py](file:///c:/Users/mhars/Downloads/PROJ%203/data_loader.py)
- Update to load `objects.pkl` (detected object tags).
- Update generator to yield `[image_features, object_features, sequence]`.

#### [NEW] [object_extractor.py](file:///c:/Users/mhars/Downloads/PROJ%203/object_extractor.py)
- Load YOLOv8 model.
- Iterate through `Images/`.
- Detect objects and save them as a multi-hot vector (or list of classes) to `objects.pkl`.

### Model
#### [MODIFY] [model.py](file:///c:/Users/mhars/Downloads/PROJ%203/model.py)
- Update `define_model` to accept `num_object_classes`.
- **Input 1**: Image Features (4096).
- **Input 2**: Object Features (80).
- **Merge**: Concatenate (Input 1 + Input 2) -> Dense -> Context Vector.
- **Decoder**: LSTM (unchanged).

### Training
#### [MODIFY] [train.py](file:///c:/Users/mhars/Downloads/PROJ%203/train.py)
- Load `objects.pkl`.
- Update data generator to feed object features.
- Retrain model.

### Inference
#### [MODIFY] [inference.py](file:///c:/Users/mhars/Downloads/PROJ%203/inference.py)
- Run YOLO on input image to get tags.
- Pass tags + features to model.

#### [MODIFY] [GUI_App/app.py](file:///c:/Users/mhars/Downloads/PROJ%203/GUI_App/app.py)
- Display detected objects in the UI.

## Verification Plan
### Automated Tests
- Verify `object_extractor.py` produces correct tags for sample images.
- Verify model accepts dual inputs.

### Manual Verification
- Check if captions mention the detected objects more accurately.

## Phase 5: Beam Search
### Goal
Implement Beam Search to generate better captions by exploring multiple probability paths instead of just the single best one.

### Changes
#### [MODIFY] [inference.py](file:///c:/Users/mhars/Downloads/PROJ%203/src/inference.py)
- Add `beam_search_predictions(model, tokenizer, photo, object_vector, beam_index=3)` function.
- It maintains top `k` sequences at each step.

#### [MODIFY] [gui/app.py](file:///c:/Users/mhars/Downloads/PROJ%203/gui/app.py)
- Add a Checkbox or Switch for "Use Beam Search".
- Call the new function when enabled.
