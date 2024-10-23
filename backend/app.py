import os
import uuid
import asyncio
import aiohttp
from flask import Flask, request, jsonify, send_file, current_app
from flask_cors import CORS
from werkzeug.utils import secure_filename
from image_processing import upscale_image
from image_analysis import analyze_image
import logging

app = Flask(__name__)
# Update CORS settings to allow requests from Vercel
CORS(app, resources={
    r"/*": {
        "origins": ["https://*.vercel.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
app.config['UPLOAD_FOLDER'] = 'static/temp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return "Welcome to Centaurion Slidr - AI-Powered Image Upscaling and Analysis"

@app.route('/upload', methods=['POST'])
def upload_files():
    logger.info('Received upload request')
    if 'files' not in request.files:
        logger.error('No file part in the request')
        return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        logger.error('No selected files')
        return jsonify({'error': 'No selected files'}), 400
    
    uploaded_files = []
    for file in files:
        if file.filename:
            try:
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                uploaded_files.append(unique_filename)
                logger.info(f'Successfully uploaded file: {unique_filename}')
            except Exception as e:
                logger.error(f'Error uploading file {file.filename}: {str(e)}')
                return jsonify({'error': f'Error uploading file {file.filename}'}), 500
    
    logger.info(f'Successfully uploaded {len(uploaded_files)} files')
    return jsonify({'filenames': uploaded_files}), 200

async def process_image(filename, input_path, output_path):
    try:
        await asyncio.to_thread(upscale_image, input_path, output_path)
        logger.info(f'Successfully processed image: {filename}')
        return output_path
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return None

@app.route('/upscale', methods=['POST'])
async def upscale():
    logger.info('Received upscale request')
    data = request.json
    filenames = data.get('filenames')
    if not filenames:
        logger.error('No filenames provided for upscaling')
        return jsonify({'error': 'No filenames provided'}), 400
    
    upscaled_files = []
    tasks = []
    for filename in filenames:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_path):
            logger.error(f'File not found: {filename}')
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        output_filename = f"upscaled_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        task = asyncio.create_task(process_image(filename, input_path, output_path))
        tasks.append(task)
    
    try:
        results = await asyncio.gather(*tasks)
        upscaled_files = [os.path.basename(path) for path in results if path]
        
        if not upscaled_files:
            logger.error('No images were successfully upscaled')
            return jsonify({'error': 'Failed to upscale images'}), 500
            
        logger.info(f'Successfully upscaled {len(upscaled_files)} files')
        return jsonify({'upscaled_filenames': upscaled_files}), 200
    except Exception as e:
        logger.error(f'Error during batch upscaling: {str(e)}')
        return jsonify({'error': 'Internal server error during upscaling'}), 500

async def analyze_image_async(original_path, upscaled_path):
    try:
        result = await asyncio.to_thread(analyze_image, original_path, upscaled_path)
        return result
    except Exception as e:
        logger.error(f"Error analyzing {original_path} and {upscaled_path}: {str(e)}")
        return None

@app.route('/analyze', methods=['POST'])
async def analyze():
    logger.info('Received analyze request')
    data = request.json
    original_filenames = data.get('original_filenames')
    upscaled_filenames = data.get('upscaled_filenames')
    
    if not original_filenames or not upscaled_filenames or len(original_filenames) != len(upscaled_filenames):
        logger.error('Invalid filenames provided for analysis')
        return jsonify({'error': 'Invalid filenames provided'}), 400
    
    analysis_results = []
    tasks = []
    for original_filename, upscaled_filename in zip(original_filenames, upscaled_filenames):
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        upscaled_path = os.path.join(app.config['UPLOAD_FOLDER'], upscaled_filename)
        
        if not os.path.exists(original_path) or not os.path.exists(upscaled_path):
            logger.error(f'One or both files not found: {original_filename}, {upscaled_filename}')
            return jsonify({'error': f'One or both files not found: {original_filename}, {upscaled_filename}'}), 404
        
        task = asyncio.create_task(analyze_image_async(original_path, upscaled_path))
        tasks.append((original_filename, upscaled_filename, task))
    
    try:
        for original_filename, upscaled_filename, task in tasks:
            result = await task
            if result:
                analysis_results.append({
                    'original_filename': original_filename,
                    'upscaled_filename': upscaled_filename,
                    'analysis': result
                })
        
        if not analysis_results:
            logger.error('No images were successfully analyzed')
            return jsonify({'error': 'Failed to analyze images'}), 500
            
        logger.info(f'Successfully analyzed {len(analysis_results)} image pairs')
        return jsonify(analysis_results), 200
    except Exception as e:
        logger.error(f'Error during batch analysis: {str(e)}')
        return jsonify({'error': 'Internal server error during analysis'}), 500

@app.route('/image/<filename>')
def get_image(filename):
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except Exception as e:
        logger.error(f'Error serving image {filename}: {str(e)}')
        return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
