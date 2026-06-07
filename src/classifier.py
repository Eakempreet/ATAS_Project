"""
classifier.py
-------------
Loads the trained EfficientNetV2-L aircraft classifier and runs inference
with Test Time Augmentation (TTA).

Responsibilities:
    - Set TF_USE_LEGACY_KERAS before any TensorFlow import
    - Load the saved .keras model from disk once at module import time
    - Accept an image file path and return a predicted aircraft class name
    - Apply TTA (N=15) using horizontal flip, vertical flip,
      brightness and contrast jitter

Used by: main.py (FastAPI endpoint)
"""

# Must be set before any TensorFlow or TF Hub import.
# The model was trained with tf_keras (Keras 2) and saved in that format.
# Without this flag, TF Hub's KerasLayer will fail to deserialize correctly
# on TF 2.13+ which defaults to Keras 3.
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


import tensorflow as tf
import tensorflow_hub as hub
from schemas import MODEL_PATHS
import numpy as np

# Load the Keras model

try:
    classifier_model = tf.keras.models.load_model(
        MODEL_PATHS['classifier'],
        custom_objects={"KerasLayer": hub.KerasLayer}
    )
    print("Classifier model loaded successfully.")
except FileNotFoundError:
    print(f"Error: model file not found at {MODEL_PATHS['classifier']}")
except Exception as e:
    print(f"Error loading classifier model: {e}")


CLASS_NAMES = [
    'A10', 'A400M', 'AG600', 'AH64', 'AKINCI', 'AV8B', 'An124', 'An22',
    'An225', 'An72', 'B1', 'B2', 'B21', 'B52', 'Be200', 'C1', 'C130',
    'C17', 'C2', 'C390', 'C5', 'CH47', 'CH53', 'CL415', 'E2', 'E7',
    'EF2000', 'EMB314', 'F117', 'F14', 'F15', 'F16', 'F18', 'F2',
    'F22', 'F35', 'F4', 'FCK1', 'H6', 'Il76', 'J10', 'J20', 'J35',
    'J36', 'J50', 'JAS39', 'JF17', 'JH7', 'KAAN', 'KC135', 'KF21',
    'KIZILELMA', 'KJ600', 'Ka27', 'Ka52', 'MQ20', 'MQ25', 'MQ28',
    'MQ9', 'Mi24', 'Mi26', 'Mi28', 'Mi8', 'Mig29', 'Mig31',
    'Mirage2000', 'NH90', 'P3', 'RQ4', 'Rafale', 'SR71', 'Su24',
    'Su25', 'Su34', 'Su47', 'Su57', 'T50', 'TB001', 'TB2', 'Tejas',
    'Tornado', 'Tu160', 'Tu22M', 'Tu95', 'U2', 'UH60', 'US2', 'V22',
    'V280', 'Vulcan', 'WZ10', 'WZ7', 'WZ9', 'X29', 'X32', 'XB70',
    'XQ58', 'Y20', 'YF23', 'Z10', 'Z19'
]


# Create a function to preprocess the image
def _process_image(image_path:str, image_size=(480, 480)):
    """
    Load and preprocess a single image from disk.
    - Reads raw bytes from the filepath
    - Decodes into an RGB tensor
    - Resizes to the target image size
    - Normalizes pixel values to [0, 1]

    Args:
        image_path: Path to the image file (string)
        image_size: Target image size

    Returns:
        Preprocessed image tensor
    """

    # Read the image as raw bytes from the filepath
    image = tf.io.read_file(image_path)

    # Decode into an RGB tensor
    image = tf.image.decode_jpeg(image, channels=3)

    # Convert pixel values from 0-255 to 0-1
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image to the desired shape
    image = tf.image.resize(image, image_size)

    return image



# Function to apply augmentation to Validation or Test images
def _tta_augment(image):
    """
    Applies random augmentations to a single image at inference time.
    Matches training augmentation exactly: flips, brightness, contrast.
    Used during TTA to generate N augmented versions of the same image.

    Args:
        image: preprocessed image tensor, shape (Hight, Width, 3 channels)

    Returns:
        Augmented image tensor, same shape as input.
    """
    
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    return image   


# Function to do prediction and apply augmentaion during predictions

def _tta_predict(model, image_path, n_augments=15):
    """
    Performs Test-Time Augmentation (TTA) prediction on a single image.

    This function loads an image, generates multiple augmented versions of it 
    (creating an on-the-fly batch), and passes the entire batch through the 
    model in a single forward pass. The final output is the average of the 
    softmax probabilities across all versions, which improves prediction robustness.

    Args:
        model (tf.keras.Model): The loaded aircraft classifier model.
        image_path (str or Path): The file path to the input image.
        n_augments (int, optional): The total number of image variations to 
                                    evaluate (1 original + N-1 augmentations). Defaults to 7.

    Returns:
        numpy.ndarray: A 1D array containing the averaged softmax probability 
        vector for the target image.
    """
    # 1. Load and preprocess the image
    image = _process_image(image_path=image_path)
    
    # 2. Build batch of N versions
    versions = [image]
    for _ in range(n_augments - 1):
        versions.append(_tta_augment(image))
        
    # 3. Stack into one batch (N, H, W, 3) - default batch size 7
    batch = tf.stack(versions, axis=0)
    
    # 4. One predict call
    # FAST INFERENCE: Call the model directly instead of .predict()
    # training=False ensures layers like Dropout and BatchNorm behave correctly for inference (Using a trained model to make predictions on new data)
    predictions = model(batch, training=False).numpy()
    
    # predictions = model.predict(batch, verbose=2)
    
    # 5. Average the N Softmax vectors
    avg_pred = np.mean(predictions, axis=0)
    
    return avg_pred 




# Function to predict the class name of the aircraft
def predict_aircraft(image_path:str):
    """_summary_

    Args:
        image_path (str): _description_
    """
    
    # Make the predictions
    average_tta_predictions = _tta_predict(model=classifier_model,
                                           image_path=image_path,
                                           n_augments=15)
    
    # Get the maximum probaility class index
    class_label_idx = np.argmax(average_tta_predictions)
    class_label = CLASS_NAMES[class_label_idx]
    
    return class_label