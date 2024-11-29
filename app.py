# https://devs.tw/post/448
import subprocess
import sys
# 自動安裝所需的套件
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"{package} 未安裝，現在進行安裝...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 安裝所需的套件
install_package("flask")
install_package("werkzeug")
install_package("Pillow")
install_package("numpy")
install_package("mtcnn")
install_package("opencv-python-headless")

from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import os
import random
from PIL import Image
import numpy as np
from mimetypes import guess_type 
from mtcnn import MTCNN
import cv2

app = Flask(__name__, static_folder='static')

detector = MTCNN()
expressions = ["happy", "angry", "sad", "neutral", "disgust", "fear", "surprise"]

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = 'expression_images'  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  

def predict_expression(image_path):
    image = Image.open(image_path).convert("L").resize((48, 48))
    image_data = np.array(image) / 255.0
    result = random.choice(expressions)  
    # result = "sad"
    return result

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
        result = predict_expression(file_path)
        
        folder_path = os.path.join(app.config['IMAGE_FOLDER'], result)
        if not os.path.exists(folder_path) or not os.listdir(folder_path):
            return jsonify({'error': f'No images found for expression: {result}'}), 404

        image_list = [
            f"/get_image/{result}/{filename}" for filename in os.listdir(folder_path)
        ]

        return jsonify({
            'result': result,
            'images': image_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # 清理之前上传的裁剪文件
        for old_file in os.listdir(app.config['UPLOAD_FOLDER']):
            if old_file.startswith("face_") and old_file.endswith(".jpg"):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], old_file))

        # 加载图片并转换为 RGB
        img = Image.open(file_path).convert('RGB')
        img = np.array(img)

        # 检测人脸
        results = detector.detect_faces(img)
        if len(results) == 0:
            return jsonify({'error': 'No faces detected in the image'}), 400

        # 仅保存第一张人脸
        if len(results) > 0:
            x, y, width, height = results[0]['box']
            x, y = max(0, x), max(0, y)

            cropped_face = img[y:y + height, x:x + width]
            face_path = os.path.join(app.config['UPLOAD_FOLDER'], "face_0.jpg")
            cv2.imwrite(face_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))

            return jsonify({
                'label': random.choice(expressions),
                'processed_face': f"/uploads/face_0.jpg"
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 獲取圖片文件
@app.route('/uploads/<filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
        image_url = f"/get_image/{expression}/{random_image}"  # 返回圖片的 URL

        return jsonify({'image_url': image_url, 'expression': expression})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render 會設定 PORT 環境變數
    app.run(host='0.0.0.0', port=port, debug=True)
