import os
import sys
import time
import psutil

# Add the project root directory to the system path so we can access 'models' and 'scripts' folders
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the absolute path to this script
sys.path.append(project_root)  # Add the parent directory (project root) to sys.path

import torch
from preprocess_audio import preprocess_audio_file

# Define a dictionary to map model names to preprocess functions (if needed for different models)
preprocess_map = {
    "model1": preprocess_audio_file,
}


# Meory + time logger
def log_mem_and_time(stage, start_time=None):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    msg = f"[DEBUG] {stage} - Memory Usage: {mem:.2f} MB"
    if start_time:
        elapsed = time.time() - start_time
        msg += f" | Elapsed: {elapsed:.2f} sec"
    print(msg)

# Predict drum hits from audio
def predict_audio(model_name, model, audio_file, device, threshold=0.4):
    start_time = time.time()
    log_mem_and_time("Start of predict_audio", start_time)
    
    preprocess_func = preprocess_map.get(model_name)

    log_mem_and_time("Before preprocessing", start_time)
    
    features = preprocess_func(audio_file)

    log_mem_and_time("After preprocessing", start_time)

    features = torch.tensor(features, dtype=torch.float16).to(device) # 16 instead of 32

    log_mem_and_time("After moving features to device", start_time)

    with torch.no_grad():
        log_mem_and_time("Before model prediction", start_time)
        output = model(features)
        log_mem_and_time("After model prediction", start_time)

    output_sigmoid = torch.sigmoid(output)
    predictions = (output_sigmoid > threshold).int()

    log_mem_and_time("End of predict_audio", start_time)

    return predictions