import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from data_loader import load_captions, clean_captions, load_objects, get_all_object_classes, objects_to_vector
from model import define_model
from sklearn.model_selection import train_test_split
import os

def create_sequences(tokenizer, max_length, desc_list, photo, object_vector, vocab_size):
    X1, X2, X3, y = list(), list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            X3.append(object_vector)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(X3), np.array(y)

def data_generator(descriptions, photos, objects, all_classes, tokenizer, max_length, vocab_size, batch_size=32):
    keys = list(descriptions.keys())
    while 1:
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i+batch_size]
            X1, X2, X3, y = list(), list(), list(), list()
            for key in batch_keys:
                if key not in photos:
                    continue
                photo = photos[key][0]
                desc_list = descriptions[key]
                
                # Get object vector
                obj_tags = objects.get(key, [])
                obj_vector = objects_to_vector(obj_tags, all_classes)
                
                in_img, in_seq, in_obj, out_word = create_sequences(tokenizer, max_length, desc_list, photo, obj_vector, vocab_size)
                
                for k in range(len(in_img)):
                    X1.append(in_img[k])
                    X2.append(in_seq[k])
                    X3.append(in_obj[k])
                    y.append(out_word[k])
            
            # Yield [Image, Object, Text], Target
            yield [[np.array(X1), np.array(X3), np.array(X2)], np.array(y)]

if __name__ == "__main__":
    print("Loading data...", flush=True)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    captions_file = os.path.join(BASE_DIR, "../data/captions.txt")
    features_file = os.path.join(BASE_DIR, "../models/features.pkl")
    objects_file = os.path.join(BASE_DIR, "../models/objects.pkl")
    tokenizer_file = os.path.join(BASE_DIR, "../models/tokenizer.pkl")
    
    mapping = load_captions(captions_file)
    mapping = clean_captions(mapping)
    
    with open(features_file, 'rb') as f:
        features = pickle.load(f)
        
    # Load objects
    if os.path.exists(objects_file):
        print("Loading object tags...")
        objects_mapping = load_objects(objects_file)
        all_object_classes = get_all_object_classes(objects_mapping)
        num_object_classes = len(all_object_classes)
        print(f"Found {num_object_classes} unique object classes.")
    else:
        print("Objects file not found! Run object_extractor.py first.")
        exit(1)
        
    mapping = {k: v for k, v in mapping.items() if k in features}
    
    # Limit for debugging
    LIMIT = None
    if LIMIT:
        image_ids = list(mapping.keys())[:LIMIT]
        print(f"Using {len(image_ids)} images.", flush=True)
    else:
        image_ids = list(mapping.keys())
        print(f"Using ALL {len(image_ids)} images.", flush=True)
    
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
    
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 34
    
    train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    train_descriptions = {k: mapping[k] for k in train_ids}
    train_features = {k: features[k] for k in train_ids}
    
    print(f"Train: {len(train_descriptions)}", flush=True)
    
    # Define model with object input
    model = define_model(vocab_size, max_length, num_object_classes)
    
    epochs = 20
    batch_size = 32 # Reduced batch size
    steps = len(train_descriptions) // batch_size
    if steps == 0: steps = 1
    
    print(f"Starting training for {epochs} epochs, {steps} steps/epoch...", flush=True)
    
    for i in range(epochs):
        generator = data_generator(train_descriptions, train_features, objects_mapping, all_object_classes, tokenizer, max_length, vocab_size, batch_size)
        progbar = 0
        epoch_loss = 0
        for j in range(steps):
            X, y = next(generator)
            loss = model.train_on_batch(X, y)
            epoch_loss += loss
            progbar += 1
            print(f"Epoch {i+1}/{epochs}, Step {j+1}/{steps}, Loss: {loss:.4f}", end='\r')
        
        avg_loss = epoch_loss / steps
        print(f"\nEpoch {i+1} complete. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        model_ep_path = os.path.join(BASE_DIR, f'../models/model_ep{i+1}.keras')
        model.save(model_ep_path)
        print(f"Saved {model_ep_path}")

    model_final_path = os.path.join(BASE_DIR, '../models/model_final.keras')
    model.save(model_final_path)
    print(f"Saved {model_final_path}")
