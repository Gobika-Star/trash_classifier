from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports]
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model
MODEL_PATH = os.path.join('model', 'trash_model.h5')
model = load_model(MODEL_PATH)

# Define class names and Tamil translations
class_names = ['Plastic', 'Paper', 'Glass', 'Metal', 'Organic', 'Other']
tamil_names = {
    'Plastic': 'பிளாஸ்டிக்',
    'Paper': 'காகிதம்',
    'Glass': 'கண்ணாடி',
    'Metal': 'உலோகம்',
    'Organic': 'உயிரியல்',
    'Other': 'மற்றவை'
}

def predict_trash(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_class, confidence = predict_trash(filepath)

        # Log to CSV
        log_data = {
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        df_new = pd.DataFrame([log_data])
        if os.path.exists('data/predictions.csv') and os.path.getsize('data/predictions.csv') > 0:
            df_new.to_csv('data/predictions.csv', mode='a', header=False, index=False)
        else:
            df_new.to_csv('data/predictions.csv', mode='w', header=True, index=False)

        # Get last 5 predictions for display
        if os.path.exists('data/predictions.csv') and os.path.getsize('data/predictions.csv') > 0:
            df = pd.read_csv('data/predictions.csv')
            last_predictions = df.tail(5).to_dict('records')
        else:
            last_predictions = []

        return render_template('result.html', filename=filename, result=predicted_class, confidence=confidence, predictions=last_predictions, tamil_names=tamil_names)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
