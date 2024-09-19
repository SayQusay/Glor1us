from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from PIL import Image
import os
import numpy as np
import cv2
import torch
import joblib
import torchvision.transforms as transforms
import torchvision.models as models

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model = models.vgg16(pretrained=False)
vgg_model.classifier = nn.Sequential(*list(vgg_model.classifier.children())[:-3])
vgg_model.load_state_dict(torch.load('vgg16_feature_extractor.pth', map_location=torch.device('cpu')))
vgg_model.to(device)  # Move model to the device (GPU or CPU)
vgg_model.eval()

pca = joblib.load('pca_model.pkl')
svm = joblib.load('svm_classifier.pkl')

# Route for the home/dashboard page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the preprocessing page
@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            image = Image.open(filepath)
            resized_image = resize_image(image)
            clahe_image = apply_clahe(resized_image)
            processed_image = apply_gaussian(clahe_image)

            processed_image.save(os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename))
            return render_template('preprocessing.html', original_image=file.filename, processed_image='processed_' + file.filename)
    return render_template('preprocessing.html')

# Route for the classification page
@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        processed_image = request.form.get('processed_image')
        if processed_image:
            picture = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], processed_image))
            processed_image_tensor = process_image(picture)

            features = extract_features(processed_image_tensor, vgg_model)
            pca_features = apply_pca(features, pca)
            prediction = predict_svm(pca_features, svm)

            # Mapping numerical prediction to class names
            numerical_to_class = {0: 'glioma', 1: 'meningioma', 2: 'nontumor', 3: 'pituitary'}
            predicted_class = numerical_to_class[prediction[0]]

            return render_template('classification.html', result=predicted_class)
    return render_template('classification.html')

# Image resizing function
def resize_image(input1, target_size=(224, 224)):
    return input1.resize(target_size)

# CLAHE function
def apply_clahe(input2):
    image_cv = np.array(input2.convert('L'))  # Convert to grayscale for CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image_cv)
    return Image.fromarray(clahe_image)

# Gaussian filter function
def apply_gaussian(input3, sigma=0.8, filter_shape=(5, 5)):
    image_cv = np.array(input3)
    gaussian_img = cv2.GaussianBlur(image_cv, filter_shape, sigma)
    return Image.fromarray(gaussian_img)

# Image preprocessing for ML models
def process_image(picture):
    process = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    picture = process(picture).to(device)
    picture = picture.unsqueeze(0)  # Add batch dimension
    return picture

# Feature extraction function
def extract_features(picture, model):
    with torch.no_grad():
        features = model(picture)
    return features.cpu().numpy().flatten()

# PCA application
def apply_pca(features, pca):
    return pca.transform([features])

# SVM prediction
def predict_svm(pca_features, svm):
    return svm.predict(pca_features)

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)

