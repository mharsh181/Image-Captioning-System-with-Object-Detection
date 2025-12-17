# Minor Project Report
## Image Captioning System with Object Detection

**Submitted by:** Harsh Mishra
**Project Type:** Minor Project (3rd Semester)

---

## 1. Abstract
This project aims to develop an advanced Image Captioning System capable of generating descriptive and grammatically correct captions for images. The system integrates **Deep Learning** techniques, specifically a **CNN-LSTM** architecture, to bridge the gap between Computer Vision and Natural Language Processing. To enhance the semantic accuracy of the captions, we integrated **Object Detection (YOLOv8)** to provide the model with explicit information about the objects present in the scene. Furthermore, **Beam Search** was implemented to improve the quality of the generated sentences during inference.

## 2. Introduction
Image Captioning is the process of generating a textual description of an image. It requires the model to understand both the visual content of the image (objects, actions, scene) and the relationships between them, and then express this understanding in natural language.
**Problem Statement:** Standard image captioning models often miss small but crucial details or generate repetitive, generic phrases.
**Solution:** We propose a dual-input model that combines visual features (from VGG16) with semantic object tags (from YOLOv8) to generate richer captions.

## 3. Methodology
Our approach uses an Encoder-Decoder architecture:

### 3.1. Feature Extraction (The "Eyes")
*   **Visual Features:** We use **VGG16**, a pre-trained Convolutional Neural Network (CNN), to extract a 4096-dimensional feature vector from the image. This captures the global visual appearance.
*   **Object Features:** We use **YOLOv8** (You Only Look Once) to detect specific object classes (e.g., "person", "dog", "car"). These detected tags are converted into a multi-hot vector and fed into the model as a secondary input.

### 3.2. Sequence Generation (The "Brain")
*   **LSTM (Long Short-Term Memory):** We use an LSTM network as the decoder. It takes the combined visual and object features as the initial state and generates the caption word by word.
*   **Word Embeddings:** We use an Embedding layer to convert words into dense vectors, capturing their semantic meaning.

### 3.3. Inference Strategy
*   **Greedy Search:** Selects the most likely word at each step. Fast but can lead to suboptimal sentences.
*   **Beam Search:** Explores the top *k* most likely sequences simultaneously. This results in more coherent and grammatically correct captions.

## 4. Implementation Details
*   **Language:** Python 3.x
*   **Deep Learning Framework:** TensorFlow / Keras
*   **Object Detection Library:** Ultralytics (YOLOv8)
*   **GUI Framework:** CustomTkinter (for a modern, dark-mode UI)
*   **Dataset:** Flickr8k (8,000 images with 5 captions each)

### 4.1. Project Structure
*   `data/`: Contains images and raw captions.
*   `models/`: Stores trained `.keras` models, `tokenizer.pkl`, and `features.pkl`.
*   `src/`: Core scripts (`train.py`, `model.py`, `inference.py`, `data_loader.py`).
*   `gui/`: The user interface application (`app.py`).

## 5. Results
The model was trained for 20 epochs on the Flickr8k dataset.
*   **BLEU Scores:** The model achieved competitive BLEU scores, indicating good overlap with human-generated reference captions.
*   **Qualitative Analysis:** The integration of YOLOv8 significantly reduced "hallucinations" (generating objects that aren't there) and improved the mention of specific objects. Beam Search further refined the sentence structure.

## 6. Future Scope
*   **Attention Mechanism:** Implementing Bahdanau Attention to allow the model to focus on specific image regions for each word.
*   **Transformer Models:** Replacing the LSTM with a Transformer (like BERT or GPT) for state-of-the-art performance.
*   **Larger Datasets:** Training on MS-COCO (300k+ images) for better generalization.

## 7. Conclusion
We successfully built and deployed a robust Image Captioning System. The addition of Object Detection and Beam Search proved to be effective strategies for improving caption quality. The final application provides a user-friendly interface for real-time caption generation.
