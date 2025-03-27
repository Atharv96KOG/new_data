from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

app = Flask(__name__)

# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.json')

# Load your trained model
model = XGBRegressor()
try:
    model.load_model(model_path)
    print("Model loaded successfully from:", model_path)
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")

def extract_features(image):
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic features
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        max_intensity = np.max(gray)
        min_intensity = np.min(gray)
        
        # Calculate histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Combine features
        features = np.array([
            mean_intensity,
            std_intensity,
            max_intensity,
            min_intensity,
            *hist[:10]  # Use first 10 histogram bins
        ])
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        raise

def predict_hb(features):
    try:
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(1, -1))
        
        # Make prediction
        predicted_hb = model.predict(features_scaled)[0]
        return round(predicted_hb, 2)
    except Exception as e:
        print(f"Error predicting Hb: {e}")
        raise

@app.route('/')
def index():
    return render_template('import.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image file
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Extract features
        features = extract_features(image)
        
        # Make prediction
        predicted_hb = predict_hb(features)
        
        # Convert image to base64 for display
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare features dictionary
        features_dict = {
            'Mean Intensity': round(float(np.mean(features[:1])), 2),
            'Standard Deviation': round(float(np.std(features[1:2])), 2),
            'Maximum Intensity': round(float(np.max(features[2:3])), 2),
            'Minimum Intensity': round(float(np.min(features[3:4])), 2)
        }
        
        return jsonify({
            'image_path': f'data:image/png;base64,{image_base64}',
            'predicted_hb': predicted_hb,
            'features': features_dict
        })
        
    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port) 
