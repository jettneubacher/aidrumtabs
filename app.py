import os
import sys
import threading
import time
from flask import Flask, request, jsonify, send_file, abort, after_this_request
from flask_cors import CORS
import torch

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 125 * 1024 * 1024
CORS(app, resources={r"/*": {"origins": "*"}})

# Set project root and ensure it's in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from predict import predict_audio
from models.model1 import DrumHitModel as Model1
from generate_pdf import generate_pdf

# Define absolute paths for uploads and models
UPLOAD_FOLDER = os.path.join(project_root, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_FOLDER = os.path.join(project_root, "models")

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model mapping
model_map = {
    "model1": Model1,
}
timeframe_map = {
    "model1": 0.116,
}

def load_model(model_name):
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not supported.")
    
    model_class = model_map[model_name]
    model_path = os.path.join(MODEL_FOLDER, f"{model_name}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    input_shape = (10, 1039)
    num_classes = 5

    model = model_class(input_shape=input_shape, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

@app.route("/predict", methods=["POST"])
def predict():
    audio_path = None
    try:
        model_name = request.form.get("model_name", "model1")
        audio_file = request.files["audio"]

        model = load_model(model_name)
        audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(audio_path)

        prediction_tensor = predict_audio(model_name, model, audio_path, device)

        timeframe_length = timeframe_map[model_name]
        predictions = [
            [round(i * timeframe_length, 2), row.tolist()]
            for i, row in enumerate(prediction_tensor)
        ]

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as cleanup_err:
                print(f"Warning: Failed to delete audio file: {cleanup_err}")

@app.route("/predict_pdf", methods=["POST"])
def predict_pdf():
    audio_path = None
    pdf_path = None
    try:
        model_name = request.form.get("model_name", "model1")
        audio_file = request.files["audio"]

        model = load_model(model_name)
        audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(audio_path)

        prediction_tensor = predict_audio(model_name, model, audio_path, device)
        timeframe_length = timeframe_map[model_name]

        predictions = [
            [round(i * timeframe_length, 2), row.tolist()]
            for i, row in enumerate(prediction_tensor)
        ]

        base_filename = os.path.splitext(audio_file.filename)[0]
        pdf_filename = f"{base_filename}_drum_tab.pdf"
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)

        output_path = generate_pdf(predictions, pdf_path)
        print(f"Generated PDF: {output_path}")

        return jsonify({
            "predictions": predictions,
            "pdf_url": pdf_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as cleanup_err:
                print(f"Warning: Failed to delete audio file: {cleanup_err}")

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Backend is running"}), 200

@app.route("/download_pdf/<filename>", methods=["GET"])
def download_pdf(filename):
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(pdf_path):
        return abort(404, description="PDF not found")

    @after_this_request
    def schedule_file_removal(response):
        def delayed_delete():
            time.sleep(2)  # Let the file finish streaming
            try:
                os.remove(pdf_path)
                print(f"Deleted PDF: {pdf_path}")
            except Exception as e:
                print(f"Warning: Failed to delete PDF: {e}")
        threading.Thread(target=delayed_delete).start()
        return response

    return send_file(pdf_path, as_attachment=True)