import os
import sys

# Add the project root directory to the system path so we can access 'models' and 'scripts' folders
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the absolute path to this script
sys.path.append(project_root)  # Add the parent directory (project root) to sys.path

import torch
from preprocess_audio import preprocess_audio_file

# Define a dictionary to map model names to preprocess functions (if needed for different models)
preprocess_map = {
    "model1": preprocess_audio_file,
}


# Predict drum hits from audio
def predict_audio(model_name, model, audio_file, device, threshold=0.4):
    preprocess_func = preprocess_map.get(model_name)
    
    features = preprocess_func(audio_file)
    features = torch.tensor(features, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(features)

    output_sigmoid = torch.sigmoid(output)
    predictions = (output_sigmoid > threshold).int()

    return predictions