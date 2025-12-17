import pickle
import numpy as np
from tensorflow.keras.models import load_model
from train import data_generator, define_model
from sklearn.model_selection import train_test_split
from data_loader import load_captions, clean_captions

def debug_training():
    print("Loading data for debug...")
    captions_file = "captions.txt"
    features_file = "features.pkl"
    tokenizer_file = "tokenizer.pkl"
    
    # Load captions
    mapping = load_captions(captions_file)
    mapping = clean_captions(mapping)
    
    # Load features
    with open(features_file, 'rb') as f:
        features = pickle.load(f)
        
    # Filter mapping
    mapping = {k: v for k, v in mapping.items() if k in features}
    print(f"Images: {len(mapping)}")
    
    # Load tokenizer
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
    
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 34
    
    # Split
    image_ids = list(mapping.keys())
    train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    
    train_descriptions = {k: mapping[k] for k in train_ids}
    train_features = {k: features[k] for k in train_ids}
    
    # Test Generator
    print("Testing generator...")
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size, batch_size=2)
    try:
        inputs, outputs = next(generator)
        print(f"Generator yielded: Inputs shape: {[i.shape for i in inputs]}, Output shape: {outputs.shape}")
    except Exception as e:
        print(f"Generator failed: {e}")
        return

    # Test Model
    print("Defining model...")
    model = define_model(vocab_size, max_length)
    
    print("Testing train_on_batch...")
    try:
        loss = model.train_on_batch(inputs, outputs)
        print(f"Train on batch successful. Loss: {loss}")
    except Exception as e:
        print(f"Train on batch failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training()
