from flask import Flask, request, render_template, jsonify, send_file
import os
import uuid
import cv2
from werkzeug.utils import secure_filename
import time

from model_handler import initialize_model
import model_handler as mh
from image_processor import image_processor
from explainability import generate_explanation

app = Flask(__name__)

# Configuration - use absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize model at startup
print("Initializing model...")
initialize_model()
print("Model ready!")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction request (browser UI)
    """
    # Check if file is present
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    if not allowed_file(file.filename):
        return render_template('index.html', error='Invalid file type')
    
    filepath = None
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Validate file
        is_valid, error_msg = image_processor.validate_file(filepath)
        if not is_valid:
            os.remove(filepath)
            return render_template('index.html', error=error_msg)
        
        # Preprocess
        image_tensor, preprocess_time, original_image = image_processor.preprocess(filepath)
        
        # Predict
        prediction_result = mh.model_handler.predict(image_tensor)
        
        # Generate explanations
        explanations = generate_explanation(
            mh.model_handler.model,
            image_tensor,
            original_image,
            top_k=3
        )
        
        # Save heatmaps
        heatmap_paths = []
        for i, exp in enumerate(explanations):
            heatmap_filename = f"{uuid.uuid4()}_heatmap_{i}.png"
            heatmap_path = os.path.join(app.config['OUTPUT_FOLDER'], heatmap_filename)
            cv2.imwrite(heatmap_path, exp['heatmap'])
            heatmap_paths.append(heatmap_filename)
        
        # Prepare results
        results = {
            'predictions': prediction_result['predictions'],
            'preprocess_time_ms': preprocess_time,
            'inference_time_ms': prediction_result['inference_time_ms'],
            'total_time_ms': preprocess_time + prediction_result['inference_time_ms'],
            'device': prediction_result['device'],
            'explanations': [
                {
                    'class_name': exp['class_name'],
                    'probability': exp['probability'],
                    'heatmap_url': f"/heatmap/{heatmap_paths[i]}"
                }
                for i, exp in enumerate(explanations)
            ]
        }
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('results.html', results=results)
    
    except Exception as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return render_template('index.html', error=f'Error processing image: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint for predictions
    
    Expected: multipart/form-data with 'file' field
    Returns: JSON with predictions and timing
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: JPG, PNG, DICOM'}), 400
    
    filepath = None
    try:
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Validate
        is_valid, error_msg = image_processor.validate_file(filepath)
        if not is_valid:
            os.remove(filepath)
            return jsonify({'error': error_msg}), 400
        
        # Process
        image_tensor, preprocess_time, _ = image_processor.preprocess(filepath)
        
        # Predict
        prediction_result = mh.model_handler.predict(image_tensor)
        
        # Clean up
        os.remove(filepath)
        
        # Return JSON response
        response = {
            'success': True,
            'predictions': prediction_result['predictions'],
            'timing': {
                'preprocess_ms': preprocess_time,
                'inference_ms': prediction_result['inference_time_ms'],
                'total_ms': preprocess_time + prediction_result['inference_time_ms']
            },
            'device': prediction_result['device']
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/heatmap/<filename>')
def get_heatmap(filename):
    """Serve generated heatmap images"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    return "File not found", 404

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': mh.model_handler is not None,
        'device': str(mh.model_handler.device) if mh.model_handler else 'N/A'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)