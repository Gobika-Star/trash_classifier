# Trash Classifier using AI/ML

A complete AI/ML project for classifying waste images into categories using TensorFlow/Keras and Flask web interface.

## Features

- **AI Model**: Transfer learning with MobileNetV2 for 6-class classification (Plastic, Paper, Glass, Metal, Organic, Other)
- **Web Interface**: Flask app with attractive UI, camera capture, and file upload
- **Prediction Logging**: CSV storage of predictions with timestamps
- **Responsive Design**: Bootstrap-based UI with green theme

## Categories

- Plastic
- Paper
- Glass
- Metal
- Organic
- Other

## Setup Instructions

1. **Clone or Download** the project.

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model** (Optional - Pre-trained model included):
   - Open `model/train_model.ipynb` in Jupyter Notebook
   - Run all cells to train and save the model
   - Model will be saved as `app/model/trash_model.h5`

5. **Run the Application**:
   ```bash
   cd app
   python app.py
   ```

6. **Access the App**:
   - Open browser: http://127.0.0.1:5000
   - Upload image or use camera to classify waste

## Project Structure

```
trash_classifier_project/
├── app/
│   ├── app.py                 # Flask application
│   ├── model/
│   │   └── trash_model.h5     # Trained model
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css      # Stylesheet
│   │   └── uploads/           # Uploaded images
│   ├── templates/
│   │   ├── index.html         # Home page
│   │   └── result.html        # Result page
│   └── data/
│       └── predictions.csv    # Prediction logs
├── model/
│   └── train_model.ipynb      # Training notebook
├── dataset/
│   ├── train/                 # Training images
│   └── test/                  # Test images
├── scripts/
│   └── train_model.py         # Alternative training script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Technologies Used

- **Backend**: Python, Flask
- **AI/ML**: TensorFlow, Keras, MobileNetV2
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Data Processing**: NumPy, Pandas, OpenCV, Pillow

## Usage

1. Upload an image of waste or capture using camera
2. Click "Classify Waste"
3. View prediction with confidence percentage
4. See last 5 predictions in table
5. Try again with new image

## Model Training

- **Dataset**: TrashNet-style with 6 categories
- **Preprocessing**: Resize to 224x224, normalize, data augmentation
- **Architecture**: MobileNetV2 base + Dense layers
- **Loss**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

## Contributing

Feel free to contribute by improving the model, UI, or adding features.

## License

This project is open-source. Developed by Gobika for AI/ML learning purposes.
