# https://devs.tw/post/448
import subprocess
import sys
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not installed, proceeding with installation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install_package("flask")
install_package("werkzeug")
install_package("Pillow")
install_package("numpy")

from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import os
import random
from PIL import Image
import numpy as np
from mimetypes import guess_type 
import torch
import torch.nn as nn  
from torchvision import models, transforms
import cv2

app = Flask(__name__, static_folder='static')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = 'expression_images'  
app.config['MODEL_FOLDER'] = os.path.join('static', 'predict_photo')
app.config['MODEL_FILE'] = 'best_model_fold7.pth'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  

# sys.path.append(os.path.join(os.path.dirname(__file__), 'static', 'predict_photo'))
# from static.predict_photo.run_model1 import run_model1
# print(run_model1())  
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_to_label = {idx: emotion for idx, emotion in enumerate(emotions)}

# def predict_expression(image_path):
#     image = Image.open(image_path).convert("L").resize((48, 48))
#     image_data = np.array(image) / 255.0
#     result = random.choice(emotions)  
#     # result = "sad"
#     return result

def get_regnet(num_classes):
    model = models.regnet_y_400mf(pretrained=True)
    for name, param in model.named_parameters():
        if "trunk_output.block1" in name or "trunk_output.block2" in name:
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
model_path = os.path.join(app.config['MODEL_FOLDER'], app.config['MODEL_FILE'])
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
num_classes = len(emotions)
model = get_regnet(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# index.html
@app.route('/')
@app.route('/index.html')
def index():
    return send_from_directory('.', 'index.html')
# video.html
@app.route('/video.html')
def video():
    return send_from_directory('.', 'video.html')
@app.route('/generate.html')
def g():
    return send_from_directory('.', 'generate.html')
@app.route('/test.html')
def test():
    return send_from_directory('.', 'test.html')

# static folder
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# photo predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': f'Could not read image at {file_path}'}), 400

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = transform_test(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        predicted_emotion = emotion_to_label[predicted.item()]
        
        folder_path = os.path.join(app.config['IMAGE_FOLDER'], predicted_emotion)
        if not os.path.exists(folder_path) or not os.listdir(folder_path):
            return jsonify({'error': f'No images found for expression: {predicted_emotion}'}), 404
        image_list = [
            f"/get_image/{predicted_emotion}/{filename}" for filename in os.listdir(folder_path)
        ]

        return jsonify({
            'result': predicted_emotion,
            'images': image_list
        })
        # return jsonify({'result': predicted_emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_image/<expression>/<filename>', methods=['GET'])
def get_image(expression, filename):
    folder_path = os.path.join(app.config['IMAGE_FOLDER'], expression)
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        mimetype, _ = guess_type(file_path)
        return send_file(file_path, mimetype=mimetype)
    return jsonify({'error': 'Image not found'}), 404

# video predict
@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        results = ["happy", "sad", "angry"] 
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_expression():
    try:
        data = request.json  
        if 'expression' not in data:
            return jsonify({'error': 'No expression provided'}), 400

        expression = data['expression']
        folder_path = os.path.join(app.config['IMAGE_FOLDER'], expression)

        if not os.path.exists(folder_path) or not os.listdir(folder_path):
            return jsonify({'error': f'No images found for expression: {expression}'}), 404

        random_image = random.choice(os.listdir(folder_path))
        image_url = f"/get_image/{expression}/{random_image}"  
        return jsonify({'image_url': image_url, 'expression': expression})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  
    
    app.run(host='0.0.0.0', port=port, debug=True)
