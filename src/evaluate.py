import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
from data_loader import load_captions, clean_captions
from tqdm import tqdm

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in tqdm(descriptions.items()):
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

if __name__ == "__main__":
    print("Loading resources...")
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    captions_file = os.path.join(BASE_DIR, "../data/captions.txt")
    features_file = os.path.join(BASE_DIR, "../models/features.pkl")
    tokenizer_file = os.path.join(BASE_DIR, "../models/tokenizer.pkl")
    model_file = os.path.join(BASE_DIR, "../models/model_final.keras")
    
    # Load data
    mapping = load_captions(captions_file)
    mapping = clean_captions(mapping)
    
    with open(features_file, 'rb') as f:
        features = pickle.load(f)
        
    mapping = {k: v for k, v in mapping.items() if k in features}
    
    # Use the same split as training
    image_ids = list(mapping.keys())
    # IMPORTANT: Must use same limit/seed as train.py to get correct test set
    LIMIT = 1000 
    if LIMIT:
        image_ids = image_ids[:LIMIT]
    
    train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    
    test_descriptions = {k: mapping[k] for k in test_ids}
    test_features = {k: features[k] for k in test_ids}
    
    print(f"Evaluating on {len(test_descriptions)} test images...")
    
    tokenizer = pickle.load(open(tokenizer_file, 'rb'))
    model = load_model(model_file)
    max_length = 34
    
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
