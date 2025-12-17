import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from data_loader import objects_to_vector, load_objects, get_all_object_classes

def extract_features(filename):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def extract_objects(filename, model):
    results = model(filename, verbose=False)
    detected_classes = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_classes.add(class_name)
    return list(detected_classes)

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, object_vector, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict with [Image, Object, Text]
        yhat = model.predict([photo, np.array([object_vector]), sequence], verbose=0)
        
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def beam_search_predictions(model, tokenizer, photo, object_vector, max_length, beam_index=3):
    start = [tokenizer.word_index['startseq']]
    
    # start_word, score
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            sequence = pad_sequences([s[0]], maxlen=max_length)
            
            # Predict with [Image, Object, Text]
            preds = model.predict([photo, np.array([object_vector]), sequence], verbose=0)
            
            # Get top k (beam_index) predictions
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                # Summing log probabilities for numerical stability
                prob += np.log(preds[0][w] + 1e-20) 
                
                temp.append([next_cap, prob])
        
        start_word = temp
        # Sort by score
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Take top k
        start_word = start_word[-beam_index:]
        
    start_word = start_word[-1][0]
    intermediate_caption = [word_for_id(i, tokenizer) for i in start_word]
    
    final_caption = []
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

if __name__ == "__main__":
    # Load resources
    print("Loading resources...")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(BASE_DIR, '../models/tokenizer.pkl')
    model_path = os.path.join(BASE_DIR, '../models/model_final.keras')
    objects_file = os.path.join(BASE_DIR, '../models/objects.pkl')
    image_dir = os.path.join(BASE_DIR, '../data/Images')
    
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    model = load_model(model_path)
    max_length = 34
    
    # Load object classes to ensure consistent mapping
    if os.path.exists(objects_file):
        objects_mapping = load_objects(objects_file)
        all_object_classes = get_all_object_classes(objects_mapping)
    else:
        print("Objects file not found. Cannot infer without class mapping.")
        exit(1)
        
    # Load YOLO
    print("Loading YOLO...")
    yolo_path = os.path.join(BASE_DIR, '../models/yolov8n.pt')
    yolo = YOLO(yolo_path)
    
    # Pick a random image from Images directory
    import random
    image_files = os.listdir(image_dir)
    
    random_image = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image)
    
    print(f"Generating caption for {random_image}...")
    
    # Extract feature
    photo = extract_features(image_path)
    
    # Extract objects
    detected_objects = extract_objects(image_path, yolo)
    print(f"Detected Objects: {detected_objects}")
    
    # Convert to vector
    obj_vector = objects_to_vector(detected_objects, all_object_classes)
    
    # Generate
    description = generate_desc(model, tokenizer, photo, obj_vector, max_length)
    print("\nGenerated Caption:")
    print(description)
    
    # Remove start/end tokens for display
    clean_desc = description.replace('startseq ', '').replace(' endseq', '')
    print(f"Clean: {clean_desc}")
