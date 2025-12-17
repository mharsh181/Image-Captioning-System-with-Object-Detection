import os
import string
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np

def load_objects(filename):
    """
    Loads detected objects from a pickle file.
    Returns a dictionary mapping image filename to a list of object tags.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_all_object_classes(objects_mapping):
    """
    Returns a sorted list of all unique object classes found in the mapping.
    """
    classes = set()
    for tags in objects_mapping.values():
        classes.update(tags)
    return sorted(list(classes))

def objects_to_vector(tags, all_classes):
    """
    Converts a list of object tags to a multi-hot vector.
    """
    vector = np.zeros(len(all_classes))
    for tag in tags:
        if tag in all_classes:
            index = all_classes.index(tag)
            vector[index] = 1
    return vector

def load_captions(filename):
    """
    Loads captions from the given file.
    Returns a dictionary mapping image filename to a list of captions.
    """
    df = pd.read_csv(filename)
    mapping = {}
    for index, row in df.iterrows():
        image_id = row['image']
        caption = row['caption']
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping

def clean_captions(mapping):
    """
    Cleans the captions:
    - Lowercase
    - Remove punctuation
    - Remove words with numbers
    - Remove single characters
    - Add <start> and <end> tokens
    """
    table = str.maketrans('', '', string.punctuation)
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.split()
            caption = [word.translate(table) for word in caption]
            caption = [word for word in caption if len(word) > 1]
            caption = [word for word in caption if word.isalpha()]
            caption = 'startseq ' + ' '.join(caption) + ' endseq'
            captions[i] = caption
    return mapping

def visualize_sample(image_dir, mapping, num_samples=3):
    """
    Saves a figure with sample images and their captions.
    """
    images = list(mapping.keys())[:num_samples]
    
    for i, image_name in enumerate(images):
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
            
        img = mpimg.imread(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        
        captions = mapping[image_name]
        title = '\n'.join(captions)
        plt.title(title)
        plt.savefig(f'sample_{i}.png')
        plt.close()
        print(f"Saved sample_{i}.png")

if __name__ == "__main__":
    captions_file = "captions.txt"
    image_dir = "Images"
    
    if not os.path.exists(captions_file):
        print(f"File not found: {captions_file}")
    else:
        print("Loading captions...")
        mapping = load_captions(captions_file)
        print(f"Loaded {len(mapping)} images.")
        
        print("Cleaning captions...")
        mapping = clean_captions(mapping)
        print("Captions cleaned.")
        
        print("Visualizing samples...")
        visualize_sample(image_dir, mapping)
