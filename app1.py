#Using for running really "Vision" Web for presentation 
#pakai def ... (): untuk tiap isi page
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from typing import Union, List, Tuple
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import pickle
import joblib
import os
import glob

#option menu di sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="BRAINS-VS",
        options = ["Beranda","Informasi","Prapemrosesan","Klasifikasi"],
        menu_icon=None,
        icons=["house","hr","diagram-3","person"],
        default_index=0,
        #orientation="horizontal",
        styles={
            "icon": {"color":"orange"},
            "nav-link":{
                "text-align:" : "left",
                "margin":"0px",
                "--hover-color":"#eee"
            }
        }
    )

#tampilan dashboard/home
def dashboard_page():
    st.markdown("<h1 style='text-align: center; color:#000000 ;'>BRAINS-VS: Analisis Responsif Tumor Otak Menggunakan Model Hybrid VGG16-SVM </h1> <h4 style='text-align: center;color: #555;'>Dibuat oleh tim Glor1us ITS </h4>", unsafe_allow_html=True)

    st.image("MRI_Machine.jpg")

    #referensi style icon
    st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    """,
    unsafe_allow_html=True
    )
    # Membuat tiga kolom
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <style>
            .hover-box1 {
                position: relative;
                text-align: center; 
                padding: 20px;          
                width: 350px;           /*set width of the box*/
                height: 250px;          /*set height of the box*/
                border: 5px solid #ddd; 
                border-radius: 10px; 
                overflow: hidden;
            }

            .hover-box1 h3 {
                margin: 0;
                z-index: 1;
            }

            .icon-container1 {
                display: flex; 
                justify-content: center; 
                align-items: center; 
                height: 60px;
            }

            .icon-container1 i {
                font-size: 25px; 
                color: orange;
            }

            .caption1 {
                position: absolute;
                top: 100%;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.9);
                font-size: 16px;
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                transition: top 0.5s ease;
                z-index: 2;
            }
            
            .caption1 h3{
                font-size: 20px;
                color: white;
                align-items: center;
                justify-content: center;
            }

            .hover-box1:hover .caption1 {
                top: 0;
            }
            </style>
            
            <div class="hover-box1">
                <div class="icon-container1">
                    <i class="fas fa-bolt"></i>
                </div>
                <h3>Klasifikasi Citra Lebih Cepat Menggunakan Model Hybrid Machine Learning</h3>
                <div class="caption1">
                    <h3>Model Hybrid VGG16-SVM dengan Akurasi Klasifikasi Sebesar 96,36% dan Durasi Komputasi Hanya 66.1 detik</h3>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <style>
            .hover-box2 {
                position: relative;
                text-align: center; 
                padding: 24px; 
                width: 300 px;
                height: 250px;
                border: 5px solid #ddd; 
                border-radius: 10px; 
                overflow: hidden;
            }

            .hover-box2 h3 {
                margin: 0;
                z-index: 1;
            }

            .icon-container2 {
                display: flex; 
                justify-content: center; 
                align-items: center; 
                height: 60px;
            }

            .icon-container2 i {
                font-size: 25px; 
                color: pink;
            }

            .caption2 {
                position: absolute;
                top: 100%;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.9);
                color: white;
                font-size: 20px;
                display: flex;
                transition: top 0.5s ease;
                z-index: 2;
            }
            
            .caption2 h3{
                font-size: 22px;
                color: white;
                align-items: center;
                justify-content: center;
            }

            .hover-box2:hover .caption2 {
                top: 0;
            }
            </style>
            
            <div class="hover-box2">
                <div class="icon-container2">
                    <i class="fas fa-brain"></i>
                </div>
                <h3>Klasifikasi Jenis Tumor Otak dan Nontumor</h3>
                <div class="caption2">
                    <ul> 
                        <h3>Jenis Tumor yang Mampu Dideteksi: </h3>
                        <li style="text-align:left;">Meningioma</li>
                        <li style="text-align:left;">Glioma</li>
                        <li style="text-align:left;">Pituitary</li>
                        <li style="text-align:left;">Nontumor</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

#halaman pre-processing
def pre_processing():
    st.markdown("<h1 style='text-align: left;custom-font {font-family:'Courier New',Courier,monospace;};color: red;'>Prapemrosesan</h1>", unsafe_allow_html=True)
    # File uploader in the sidebar
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([0.4, 0.6], gap='medium')
        with col1:
            st.markdown("<h4 style='color: orange;'>Metode</h4>", unsafe_allow_html=True)
            Preproc = st.checkbox("Prapemrosesan Citra")
            apply = st.button("Terapkan", type="secondary")
            
            # Image Comparisson Result
            with col2:
                st.markdown("<h4 style='color: orange;'>Hasil</h4>", unsafe_allow_html=True)
                
                processed_img = image
                if apply:
                    if Preproc:
                        #st.title("Resize")
                        resized_img = resize_image(image)
                        clahe_img = apply_clahe(resized_img)
                        processed_img = apply_gaussian(clahe_img)
                    else:
                        processed_img = image     #If image not resized
                # Save processed image to session state
                st.session_state['processed_img'] = processed_img
                            
                image_comparison(
                    img1 = processed_img,
                    img2 = image,
                    label1 = "Prapemrosesan",
                    label2 = "Input",
                    width=400,
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True)
# RESIZING
def resize_image(input1, target_size=(224, 224)):
    resized_img = input1.resize(target_size)
    return resized_img

#CLAHE
def apply_clahe(input2):
    # Convert PIL image to OpenCV format (numpy array)
    image_cv = np.array(input2.convert('L'))               #convert to grayscale for CLAHE
    
    # Create CLAHE object and apply it to image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(image_cv)
    
    # Convert back to PIL image format
    return Image.fromarray(clahe_image)

#GAUSSIAN
def generate_gaussian_filter(sigma: float, filter_shape: tuple):
    m,n = filter_shape
    m_half = m//2
    n_half = n//2
    
    gaussian_filter = np.zeros((m,n), np.float32)
    
    for y in range (-m_half, m_half+1):
        for x in range (-n_half, n_half+1):
            normal = 1/(2.0 * np.pi * sigma**2.0)
            exp_term = np.exp(-(x**2.0 + y**2.0)/(2.0 * sigma**2.0))
            gaussian_filter[y+m_half, x+n_half] = normal * exp_term
            
    # Normalize the filter to ensure the sum is 1
    gaussian_filter /= gaussian_filter.sum()
    return gaussian_filter

def apply_gaussian(input3, sigma=0.8, filter_shape=(5,5)):
    image_cv = np.array(input3)
    if len(image_cv.shape) == 2:
        gaussian_img = cv2.GaussianBlur(image_cv, filter_shape, sigma)
    else:
        gaussian_img = cv2.GaussianBlur(image_cv, filter_shape, sigma)
    return Image.fromarray(gaussian_img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load Models
vgg_model = models.vgg16(pretrained=False)
vgg_model.classifier = nn.Sequential(*list(vgg_model.classifier.children())[:-3])
vgg_model.load_state_dict(torch.load('vgg16_feature_extractor.pth', map_location=torch.device('cpu')))
vgg_model.to(device) #move model to the device (GPU or CPU)
vgg_model.eval()

pca = joblib.load('pca_model.pkl')
svm = joblib.load('svm_classifier.pkl')

# Process Image for ML
def process_image(picture):
    picture = picture.convert('RGB')    # Ensure the image is in RGB mode
    process = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    picture = process(picture).to(device) # Move the tensor to the device
    picture = picture.unsqueeze(0)  # Add batch dimension
    return picture

# Feature extraction using VGG16
def extract_features(picture, model):
    with torch.no_grad():
        features = model(picture)
    return features.cpu().numpy().flatten() # Move to CPU and convert to numpy array

# Apply PCA
def apply_pca(features, pca):
    pca_features = pca.transform([features])
    return pca_features

# Prediction using SVM
def predict_svm(pca_features, svm):
    prediction = svm.predict(pca_features)
    return prediction

# Classification Page
def classification():
    st.markdown("<h1 style='text-align: left;custom-font {font-family:'Courier New',Courier,monospace;};color: red;'>Klasifikasi</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([0.5, 0.5], gap='medium')
    with col1:
        st.markdown("<h4 style='color: orange;'>Citra</h4>", unsafe_allow_html=True)
        if 'processed_img' in st.session_state:
            st.image(st.session_state['processed_img'], caption='Processed Image', use_column_width=True)
        else:
            st.warning("Silahkan proses citra terlebih dahulu pada halaman prapemrosesan")
            return
        
    with col2:
        st.markdown("<h4 style='color: orange;'>Hasil</h4>", unsafe_allow_html=True)
        apply = st.button("Terapkan", type="secondary")
        if apply:
            processed_image = process_image(st.session_state['processed_img'])  # Preprocess image for ML
            features = extract_features(processed_image, vgg_model)             # Extract features using VGG16
            pca_features = apply_pca(features, pca)
            prediction = predict_svm(pca_features, svm)
            
            # Mapping numerical prediction to class names
            numerical_to_class = {0: 'glioma', 1: 'meningioma', 2: 'nontumor', 3: 'pituitary'}
            predicted_class = numerical_to_class[prediction[0]]
            
            with st.expander("Penjelasan Hasil Klasifikasi"):
                if predicted_class == 'glioma':
                    st.markdown (
                        """
                        ***GLIOMA***
                        - **Asal: Mutasi sel saraf lial, termasuk astrosit, oligodendorsit, dan sel ependymal**
                        """
                    )
                elif predicted_class == 'meningioma':
                    st.markdown (
                        """
                        ***MENINGIOMA***
                        - **Asal: Meninges yang menutupi otak dan sumsum tulang belakang**
                        """
                    )
                elif predicted_class == 'nontumor':
                    st.markdown (
                        """
                        ***OTAK SEHAT***
                        """
                    )
                elif predicted_class == 'pituitary':
                    st.markdown (
                        """
                        ***PITUITARI***
                        - **Asal: Mutasi sel kelenjar pituitari**
                        """
                    )

# data = {
#         'Jenis Tumor': ['Glioma', 'Meningioma', 'Pituitari', 'Lainnya'],
#         'Jumlah Kasus': [40000, 30000, 10000, 5000]
#     }
# df = pd.DataFrame(data)

def set_png_as_page_bg(png_file):
    page_bg_img = '''
    <style>
    body{
    background-image: url("https://img.freepik.com/free-photo/modern-hospital-machinery-illuminates-blue-mri-scanner-generated-by-ai_188544-44420.jpg?t=st=1717934818~exp=1717938418~hmac=739236247ba1b14c749b3e3dea9f4898d3ce83e98353a01f6537adbb396061f6&w=1380");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_png_as_page_bg("modern-hospital-machinery-illuminates-blue-mri-scanner-generated-by-ai.jpg")

# About Page
def about():
        st.markdown("<h1 style='text-align: left;custom-font {font-family:'Courier New',Courier,monospace;};color: red;'>Informasi Tumor Otak</h1>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Seputar Tumor Otak", "Statistik Tumor Otak", "Jenis Tumor"])

        # Tab 1: Data Global dan Data Lokal
        with tab1:
            # st.header("Data Global dan Data Lokal")
            st.header("Apa itu Tumor Otak?")
            st.write(f"Tumor Otak dan Sistem Saraf Terpusat merupakan kondisi perkembangan jaringan atau sel-sel abnormal di dalam otak dan sumsum tulang belakang")

            st.write("### Penyebab Tumor Otak")
            st.write("#### Faktor Lingkungan:")
            st.write("""
            1. Merokok
            2. Konsumsi alkohol yang berlebihan
            3. Pola makan yang buruk
            4. Kurang olahraga
            5. Paparan sinar matahari yang berlebihan
            6. Perilaku seksual yang meningkatkan paparan terhadap virus tertentu
            7. Konsumsi obat-obatan bebas
            8. Paparan luas terhadap bahan kimia industri yang berbahaya""")
            
            st.write("#### Faktor Genetik:")
            st.write("""
            1. Perubahan gen di dalam sel tubuh
            2. Kadar hormon yang tidak normal dalam aliran darah
            3. Sistem kekebalan tubuh yang lemah dan paparan terhadap karsinogen""", unsafe_allow_html=True)
        
        # Tab 2: Statistik Kematian akibat Tumor Otak
        with tab2:
            # st.header("Data Global dan Data Lokal")
            st.header("Data Global")
            st.write(f"Pada tahun 2022, terdapat 321.731 kasus tumor otak di seluruh dunia, di mana 248.500 di antaranya mengalami kematian (Global Cancer Statistics 2022).")

            st.header("Data Lokal (Indonesia)")
            st.write("Menurut Ananditha, T (2021), estimasi kasus angka kejadian dan kematian di Indonesia mencapai 6.337/5.405 yang disebabkan oleh 3 hal yaitu radiasi, rokok, dan genetik.")

            # Bar chart untuk kasus dan kematian di Indonesia
            fig2, ax2 = plt.subplots()
            labels = ['Kasus', 'Kematian']
            values = [6337, 5405]
            bars = ax2.bar(labels, values, color=['#ff9999', '#66b3ff'])

            for bar in bars:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')  # va: vertical alignment

            st.pyplot(fig2)

        # Tab 3: Jenis Tumor
        with tab3:
            st.header("Jenis Tumor Otak")
            st.write(f"Tumor otak digolongkan menjadi dua kelompok, yaitu tumor otak primer yang berasal dari jaringan otak maupun area dekatnya (sel glial atau non-glial), serta tumor otak metastasis (tumor ganas). Pada orang dewasa, tumor ganas yang paling umum menyerang adalah glioma, sedangkan tumor jinak adalah meningioma dan pituitari (Fan, et al., 2022).")
            st.write("### 1. Meningioma")
            st.write(f"Meningioma merupakan tumor yang berasal dari meninges yang menutupi otak dan sumsum tulang belakang. Meningioma termasuk pada jenis tumor paling umum, yaitu sebesar {36.8}% dari total CNS tumor.")
            # Pie chart untuk proporsi tumor otak
            labels = 'Meningioma', 'Jenis tumor lainnya'
            sizes = [36.8, 100-36.8]
            colors = ['#ff9999','#66b3ff']
            explode = (0.1, 0)

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            st.pyplot(fig1)
            
            st.write("### 2. Glioma")
            st.write(f"Glioma merupakan tumor  neoplasma CNS yang berasal dari sel saraf lial, termasuk astrosit, oligodendrosit, dan sel ependymal. Jenis yang paling umum dari jenis yang paling umum dari glioma tingkat rendah adalah astrositoma dan oligodendro glioma, yang cenderung tumbuh lambat dan lamban, sedangkan glioma tingkat tinggi termasuk astrositoma anaplastik dan glioblastoma dan jauh lebih agresif dan keganasan yang tumbuh dengan cepat.")
            st.write("### 3. Pituitari")
            st.write(f"Pituitari merupakan tumor yang terjadi pada daerah sellar orang dewasa. Pituitari termasuk jenis tumor yang banyak diderita sebesar {15}% dari jumlah keseluruhan CNS. Tumor ini umumnya dikelompokkan berdasarkan ukurannya, yaitu microadenomas dan macroadenomas, serta berdasarkan tipe histologinya.")
            st.image("hasil.jpg")

            

if selected == "Beranda":
    dashboard_page()
if selected == "Prapemrosesan":
    pre_processing()
if selected == "Klasifikasi":
    classification()
if selected == "Informasi":
    about()



