import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from data_loader import load_captions, clean_captions

def create_tokenizer(captions_mapping):
    """
    Fits a tokenizer on the list of captions.
    """
    all_captions = []
    for key in captions_mapping:
        for caption in captions_mapping[key]:
            all_captions.append(caption)
            
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

def get_max_length(captions_mapping):
    """
    Calculates the maximum length of captions.
    """
    all_captions = []
    for key in captions_mapping:
        for caption in captions_mapping[key]:
            all_captions.append(caption)
            
    return max(len(c.split()) for c in all_captions)

if __name__ == "__main__":
    captions_file = "captions.txt"
    tokenizer_file = "tokenizer.pkl"
    
    print("Loading and cleaning captions...")
    mapping = load_captions(captions_file)
    mapping = clean_captions(mapping)
    
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(mapping)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")
    
    max_length = get_max_length(mapping)
    print(f"Max Caption Length: {max_length}")
    
    # Save tokenizer
    with open(tokenizer_file, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {tokenizer_file}")
