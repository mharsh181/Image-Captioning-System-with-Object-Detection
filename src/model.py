from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model

def define_model(vocab_size, max_length, num_object_classes=None):
    """
    Defines the CNN-LSTM model for image captioning.
    Supports optional object detection input.
    
    Args:
        vocab_size: Size of the vocabulary.
        max_length: Maximum length of the caption sequence.
        num_object_classes: Number of object classes (optional).
        
    Returns:
        A Keras Model instance.
    """
    # Feature Extractor (Encoder) - Image
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence Processor (Decoder) - Text
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Object Feature Extractor (Optional)
    if num_object_classes:
        inputs3 = Input(shape=(num_object_classes,))
        obj1 = Dense(256, activation='relu')(inputs3)
        
        # Merge Image + Objects first to get Visual Context
        visual_context = add([fe2, obj1])
        
        # Merge Visual Context + Text
        decoder1 = add([visual_context, se3])
        inputs_list = [inputs1, inputs3, inputs2]
    else:
        # Standard Merge
        decoder1 = add([fe2, se3])
        inputs_list = [inputs1, inputs2]
    
    # Decoder (Output)
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Tie it together
    model = Model(inputs=inputs_list, outputs=outputs)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    print(model.summary())
    return model
