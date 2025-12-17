import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from data_loader import load_captions, clean_captions
from sklearn.model_selection import train_test_split
import os

def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size, batch_size=32):
    keys = list(descriptions.keys())
    while 1:
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i+batch_size]
            X1, X2, y = list(), list(), list()
            for key in batch_keys:
                if key not in photos:
                    continue
                photo = photos[key][0]
                desc_list = descriptions[key]
                in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
                
                for k in range(len(in_img)):
                    X1.append(in_img[k])
                    X2.append(in_seq[k])
                    y.append(out_word[k])
            
            yield [[np.array(X1), np.array(X2)], np.array(y)]

if __name__ == "__main__":
    print("Loading data...", flush=True)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    captions_file = os.path.join(BASE_DIR, "../data/captions.txt")
    features_file = os.path.join(BASE_DIR, "../models/features.pkl")
    tokenizer_file = os.path.join(BASE_DIR, "../models/tokenizer.pkl")
    model_file = os.path.join(BASE_DIR, "../models/model_final.keras")
    
    mapping = load_captions(captions_file)
    mapping = clean_captions(mapping)
    
    with open(features_file, 'rb') as f:
        features = pickle.load(f)
        
    mapping = {k: v for k, v in mapping.items() if k in features}
    
    # Full dataset
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
    
    print(f"Loading existing model from {model_file}...", flush=True)
    model = load_model(model_file)
    
    start_epoch = 20
    end_epoch = 50
    batch_size = 32
    steps = len(train_descriptions) // batch_size
    if steps == 0: steps = 1
    
    print(f"Continuing training from epoch {start_epoch+1} to {end_epoch}...", flush=True)
    
    for i in range(start_epoch, end_epoch):
        generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size, batch_size)
        progbar = 0
        epoch_loss = 0
        for j in range(steps):
            X, y = next(generator)
            loss = model.train_on_batch(X, y)
            epoch_loss += loss
            progbar += 1
            print(f"Epoch {i+1}/{end_epoch}, Step {j+1}/{steps}, Loss: {loss:.4f}", end='\r')
        
        avg_loss = epoch_loss / steps
        print(f"\nEpoch {i+1} complete. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        model_ep_path = os.path.join(BASE_DIR, f'../models/model_ep{i+1}.keras')
        model.save(model_ep_path)
        print(f"Saved {model_ep_path}")

    model_final_path = os.path.join(BASE_DIR, '../models/model_final.keras')
    model.save(model_final_path)
    print(f"Saved {model_final_path}")
