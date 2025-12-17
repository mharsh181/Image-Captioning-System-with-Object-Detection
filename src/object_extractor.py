import os
import pickle
from ultralytics import YOLO
from tqdm import tqdm

def extract_objects(directory, output_file, limit=None):
    """
    Detects objects in images using YOLOv8 and saves them to a pickle file.
    Returns a dictionary mapping image filename to a list of detected object class names.
    """
    # Load YOLOv8 model (pre-trained on COCO dataset)
    print("Loading YOLOv8 model...")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/yolov8n.pt')
    model = YOLO(model_path)  # 'n' for nano (fastest), 's' for small, etc.
    
    objects_dict = {}
    files = os.listdir(directory)
    
    if limit:
        files = files[:limit]
        
    print(f"Processing {len(files)} images...")
    
    # Process images in batches for efficiency (YOLO supports batch inference)
    # But for simplicity and progress tracking, we'll do one by one or small batches
    
    for filename in tqdm(files):
        filepath = os.path.join(directory, filename)
        
        try:
            # Run inference
            results = model(filepath, verbose=False)
            
            # Extract class names
            detected_classes = set() # Use set to avoid duplicates per image
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    detected_classes.add(class_name)
            
            # Store as list
            objects_dict[filename] = list(detected_classes)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            objects_dict[filename] = []

    # Save to file
    with open(output_file, 'wb') as f:
        pickle.dump(objects_dict, f)
    print(f"Saved object tags for {len(objects_dict)} images to {output_file}")
    return objects_dict

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, '../data/Images')
    output_file = os.path.join(BASE_DIR, '../models/objects.pkl')
    
    # Set limit for testing, None for full dataset
    LIMIT = None
    
    if os.path.exists(output_file):
        print(f"Objects file {output_file} already exists.")
        # Optional: Ask user if they want to overwrite? For now, we assume if it exists we might skip or overwrite.
        # Let's overwrite to be safe as this is a new phase.
        print("Overwriting...")
        extract_objects(image_dir, output_file, limit=LIMIT)
    else:
        extract_objects(image_dir, output_file, limit=LIMIT)
