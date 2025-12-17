import os
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tqdm import tqdm

def extract_features(directory, limit=None):
    """
    Extracts features from each image in the directory using VGG16.
    Returns a dictionary mapping image filename to features.
    """
    # Load the model
    model = VGG16()
    # Remove the last layer (classification layer)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print(model.summary())
    
    features = {}
    files = os.listdir(directory)
    if limit:
        files = files[:limit]
        
    for name in tqdm(files):
        filename = os.path.join(directory, name)
        # Load the image
        image = load_img(filename, target_size=(224, 224))
        # Convert the image pixels to a numpy array
        image = img_to_array(image)
        # Reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Prepare the image for the VGG model
        image = preprocess_input(image)
        # Get features
        feature = model.predict(image, verbose=0)
        # Store feature
        image_id = name # Keep the full filename as ID to match captions.txt
        features[image_id] = feature
        
    return features

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(BASE_DIR, '../data/Images')
    features_file = os.path.join(BASE_DIR, '../models/features.pkl')
    
    # Set a limit for testing/development. Set to None for full dataset.
    LIMIT = None 
    
    if os.path.exists(features_file):
        print(f"Features file {features_file} already exists.")
    else:
        if LIMIT:
            print(f"Extracting features (limit={LIMIT})...")
        else:
            print("Extracting features for ALL images...")
            
        features = extract_features(directory, limit=LIMIT)
        print(f"Extracted features for {len(features)} images.")
        # Save to file
        with open(features_file, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features saved to {features_file}")
