from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import os
import random
from PIL import Image
import numpy as np
from mimetypes import guess_type 

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = 'expression_images'  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  

def predict_expression(image_path):
    image = Image.open(image_path).convert("L").resize((48, 48))
    image_data = np.array(image) / 255.0
    expressions = ["happy", "angry", "sad", "neutral", "disgust", "fear", "surprise"]
    result = random.choice(expressions)  
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

if __name__ == '__main__':
    app.run(debug=True)
