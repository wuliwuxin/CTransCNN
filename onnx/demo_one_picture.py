import onnxruntime as ort
import numpy as np
from PIL import Image
import time
import json

def load_class_indices(json_path):
    """Loads class indices from a JSON file."""
    with open(json_path, 'r') as f:
        class_indices = json.load(f)
    return class_indices

def load_image(image_path, size=(224, 224)):
    """Loads and preprocesses the image for prediction."""
    image = Image.open(image_path)
    image = image.resize(size).convert('RGB')
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to 0-1

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_array = (image_array - mean) / std  # Normalize using ImageNet standards

    image_array = np.transpose(image_array, [2, 0, 1])  # Change to CHW format
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict(image_path, model_path):
    """Loads the ONNX model, runs the prediction, and processes the output."""
    session = ort.InferenceSession(model_path)
    image = load_image(image_path)
    
    input_name = session.get_inputs()[0].name  # Get the input name for the model
    output_name = session.get_outputs()[0].name  # Get the output name for the model
    
    start_time = time.time()
    scores = session.run([output_name], {input_name: image})[0]
    time_cost = time.time() - start_time
    
    return scores, time_cost

def run_example(image_path, model_path, class_indices_path):
    """Executes the model prediction and prints the results."""
    class_indices = load_class_indices(class_indices_path)
    scores, time_cost = predict(image_path, model_path)

    threshold = 0.5

    print("Results:")
    for idx, score in enumerate(scores[0]):
        if score > threshold:
            print(class_indices[str(idx)])  # Access class name using str(idx)

    print("Inference time: {:.4f} seconds".format(time_cost))
# 1.2.826.0.1.3680043.8.498.41477985046369504069935954327112429508.jpg CVC-Borderline CVC-Normal

model_path = 'onnx/CTransCNN_chest11.onnx'
image_path = 'onnx/1.2.826.0.1.3680043.8.498.41477985046369504069935954327112429508.jpg'
class_indices = 'onnx/Chest11_Label.json'

run_example(image_path, model_path,class_indices)
