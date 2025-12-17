import os
import pickle
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from ultralytics import YOLO

# Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ImageCaptionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Image Captioning System (with Object Detection)")
        self.geometry("1000x800")
        
        # Load Resources
        self.status_label = ctk.CTkLabel(self, text="Loading models... Please wait.", font=("Roboto", 16))
        self.status_label.pack(pady=20)
        self.update() # Force update to show label
        
        self.load_resources()
        
        self.status_label.destroy() # Remove loading label

        # UI Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        self.header = ctk.CTkLabel(self, text="Image Caption Generator + Object Detection", font=("Roboto", 24, "bold"))
        self.header.grid(row=0, column=0, pady=20, sticky="ew")

        # Main Content Area
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Image Display
        self.image_label = ctk.CTkLabel(self.main_frame, text="No Image Selected", font=("Roboto", 16))
        self.image_label.grid(row=0, column=0, pady=20)

        # Controls Area
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure((0, 1), weight=1)

        self.upload_btn = ctk.CTkButton(self.controls_frame, text="Upload Image", command=self.upload_image, font=("Roboto", 14))
        self.upload_btn.grid(row=0, column=0, padx=10, pady=20)

        self.generate_btn = ctk.CTkButton(self.controls_frame, text="Generate Caption", command=self.generate_caption, font=("Roboto", 14), state="disabled")
        self.generate_btn.grid(row=0, column=1, padx=10, pady=20)
        
        self.beam_search_var = ctk.BooleanVar(value=False)
        self.beam_search_switch = ctk.CTkSwitch(self.controls_frame, text="Use Beam Search", variable=self.beam_search_var, font=("Roboto", 12))
        self.beam_search_switch.grid(row=0, column=2, padx=10, pady=20)

        # Result Area
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.result_frame.grid_columnconfigure(0, weight=1)
        
        self.objects_label = ctk.CTkLabel(self.result_frame, text="", font=("Roboto", 14), text_color="cyan", wraplength=900)
        self.objects_label.grid(row=0, column=0, pady=(10, 5))
        
        self.result_label = ctk.CTkLabel(self.result_frame, text="", font=("Roboto", 18, "bold"), wraplength=900)
        self.result_label.grid(row=1, column=0, pady=(5, 20))

        # Footer
        self.footer = ctk.CTkLabel(self, text="MADE BY HARSH MISHRA", font=("Roboto", 12, "bold"), text_color="gray")
        self.footer.grid(row=4, column=0, pady=10)

        self.current_image_path = None

    def load_resources(self):
        try:
            print("Loading Tokenizer...")
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            tokenizer_path = os.path.join(BASE_DIR, '../models/tokenizer.pkl')
            model_path = os.path.join(BASE_DIR, '../models/model_final.keras')
            objects_file = os.path.join(BASE_DIR, '../models/objects.pkl')
            
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load Object Classes
            if os.path.exists(objects_file):
                with open(objects_file, 'rb') as f:
                    objects_mapping = pickle.load(f)
                
                classes = set()
                for tags in objects_mapping.values():
                    classes.update(tags)
                self.all_object_classes = sorted(list(classes))
                print(f"Loaded {len(self.all_object_classes)} object classes.")
            else:
                print("Warning: objects.pkl not found.")
                self.all_object_classes = []

            print("Loading Caption Model...")
            self.model = load_model(model_path)
            
            print("Loading VGG16 Model...")
            vgg = VGG16()
            self.vgg_model = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
            
            print("Loading YOLO Model...")
            self.yolo = YOLO('yolov8n.pt')
            
            self.max_length = 34
            print("Resources loaded successfully.")
        except Exception as e:
            print(f"Error loading resources: {e}")
            self.status_label.configure(text=f"Error loading resources: {e}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.current_image_path = file_path
            
            # Display Image
            img = Image.open(file_path)
            # Resize for display
            img.thumbnail((400, 400))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            
            self.image_label.configure(image=ctk_img, text="")
            self.generate_btn.configure(state="normal")
            self.result_label.configure(text="")
            self.objects_label.configure(text="")

    def extract_features(self, filename):
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = self.vgg_model.predict(image, verbose=0)
        return feature

    def extract_objects(self, filename):
        results = self.yolo(filename, verbose=False)
        detected_classes = set()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.yolo.names[class_id]
                detected_classes.add(class_name)
        return list(detected_classes)

    def objects_to_vector(self, tags):
        vector = np.zeros(len(self.all_object_classes))
        for tag in tags:
            if tag in self.all_object_classes:
                index = self.all_object_classes.index(tag)
                vector[index] = 1
        return vector

    def word_for_id(self, integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def generate_desc(self, photo, object_vector):
        in_text = 'startseq'
        for i in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            
            # Predict with [Image, Object, Text]
            yhat = self.model.predict([photo, np.array([object_vector]), sequence], verbose=0)
            
            yhat = np.argmax(yhat)
            word = self.word_for_id(yhat)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        return in_text

    def beam_search_predictions(self, photo, object_vector, beam_index=3):
        start = [self.tokenizer.word_index['startseq']]
        start_word = [[start, 0.0]]
        
        while len(start_word[0][0]) < self.max_length:
            temp = []
            for s in start_word:
                sequence = pad_sequences([s[0]], maxlen=self.max_length)
                preds = self.model.predict([photo, np.array([object_vector]), sequence], verbose=0)
                word_preds = np.argsort(preds[0])[-beam_index:]
                
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += np.log(preds[0][w] + 1e-20)
                    temp.append([next_cap, prob])
            
            start_word = temp
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            start_word = start_word[-beam_index:]
            
        start_word = start_word[-1][0]
        intermediate_caption = [self.word_for_id(i) for i in start_word]
        
        final_caption = []
        for i in intermediate_caption:
            if i != 'endseq':
                final_caption.append(i)
            else:
                break
        
        return ' '.join(final_caption[1:])

    def generate_caption(self):
        if not self.current_image_path:
            return
        
        self.result_label.configure(text="Generating caption...")
        self.objects_label.configure(text="Detecting objects...")
        self.update()
        
        try:
            # 1. Extract Image Features
            photo = self.extract_features(self.current_image_path)
            
            # 2. Extract Objects
            detected_objects = self.extract_objects(self.current_image_path)
            obj_text = "Detected: " + ", ".join(detected_objects) if detected_objects else "Detected: None"
            self.objects_label.configure(text=obj_text)
            self.update()
            
            # 3. Convert Objects to Vector
            obj_vector = self.objects_to_vector(detected_objects)
            
            # 4. Generate Caption
            if self.beam_search_var.get():
                print("Using Beam Search...")
                caption = self.beam_search_predictions(photo, obj_vector)
            else:
                print("Using Greedy Search...")
                caption = self.generate_desc(photo, obj_vector)
            
            # Clean caption
            clean_caption = caption.replace('startseq ', '').replace(' endseq', '')
            clean_caption = clean_caption.capitalize()
            
            self.result_label.configure(text=clean_caption)
        except Exception as e:
            self.result_label.configure(text=f"Error: {e}")
            print(e)

if __name__ == "__main__":
    app = ImageCaptionApp()
    app.mainloop()
