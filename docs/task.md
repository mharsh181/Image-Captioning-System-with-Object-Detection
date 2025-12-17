# Image Captioning Model - Flickr8k

- [/] Data Loading- [x] Model Architecture <!-- id: 12 -->
    - [x] Define the Image Encoder (Dense layer from features) <!-- id: 13 -->
    - [x] Define the Caption Decoder (Embedding +- [x] Training <!-- id: 16 -->
    - [x] Compile the model <!-- id: 17 -->
    - [x] Train the model <!-- id: 18 -->
    - [x] Monitor loss and accuracy <!-- id: 19 -->
- [x] Evaluation and Inference <!-- id: 20 -->
    - [x] Implement greedy search or beam search for caption generation <!-- id: 21 -->
    - [x] Evaluate on test images (BLEU score optional but good) <!-- id: 22 -->
    - [x] Generate captions for new images <!-- id: 23 -->

# Phase 2: Evaluation and Scaling
- [x] Evaluation Metrics <!-- id: 24 -->
    - [x] Implement BLEU score calculation (BLEU-1 to BLEU-4) <!-- id: 25 -->
    - [x] Evaluate current model on test set <!-- id: 26 -->
- [x] Full Dataset Training <!-- id: 27 -->
    - [x] Refactor `feature_extractor.py` to process all images <!-- id: 28 -->
    - [x] Refactor `train.py` to train on full dataset <!-- id: 29 -->
    - [x] Train model on full dataset (requires time/GPU) <!-- id: 30 -->
    - [x] Continue training to 50 epochs <!-- id: 39 -->

# Phase 3: GUI Application
- [x] Setup <!-- id: 31 -->
    - [x] Create `GUI_App` directory <!-- id: 32 -->
    - [x] Copy trained model and tokenizer <!-- id: 33 -->
- [x] Implementation <!-- id: 34 -->
    - [x] Create `app.py` with `customtkinter` <!-- id: 35 -->
    - [x] Implement image upload and display <!-- id: 36 -->
    - [x] Integrate caption generation logic <!-- id: 37 -->
    - [x] Add footer "MADE BY HARSH MISHRA" <!-- id: 38 -->
    - [x] Reorganize project structure (data, models, src, gui) <!-- id: 51 -->

# Phase 4: Object Detection Integration
- [ ] Preparation <!-- id: 40 -->
    - [x] Install `ultralytics` (YOLO) <!-- id: 41 -->
    - [x] Create `object_extractor.py` to extract object tags <!-- id: 42 -->
    - [x] Extract objects for all images <!-- id: 43 -->
- [ ] Model Upgrade <!-- id: 44 -->
    - [x] Update `data_loader.py` to load object tags <!-- id: 45 -->
    - [x] Update `model.py` to accept Object Input (Dual Input or Concatenation) <!-- id: 46 -->
    - [x] Update `train.py` to feed object data <!-- id: 47 -->
- [ ] Training & Verification <!-- id: 48 -->
    - [x] Retrain model (Phase 4) <!-- id: 49 -->
    - [x] Update GUI to show detected objects <!-- id: 50 -->

# Phase 5: Advanced Improvements
- [ ] Beam Search Implementation <!-- id: 52 -->
    - [x] Implement `beam_search_predictions` in `inference.py` <!-- id: 53 -->
    - [x] Update GUI to add "Beam Search" toggle <!-- id: 54 -->
    - [/] Verify improved caption quality <!-- id: 55 -->
