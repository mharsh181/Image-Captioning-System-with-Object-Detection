import pickle
import os
import numpy as np

def check_data():
    print("Checking features.pkl...")
    if not os.path.exists("features.pkl"):
        print("features.pkl does not exist.")
        return
    
    try:
        with open("features.pkl", "rb") as f:
            features = pickle.load(f)
        print(f"Features loaded. Count: {len(features)}")
        # Check one feature
        k = list(features.keys())[0]
        print(f"Feature shape: {features[k].shape}")
    except Exception as e:
        print(f"Error loading features.pkl: {e}")

    print("\nChecking tokenizer.pkl...")
    if not os.path.exists("tokenizer.pkl"):
        print("tokenizer.pkl does not exist.")
        return
        
    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        print(f"Tokenizer loaded. Vocab size: {len(tokenizer.word_index) + 1}")
    except Exception as e:
        print(f"Error loading tokenizer.pkl: {e}")

if __name__ == "__main__":
    check_data()
