import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib

# Ladda den sparade modellen
@st.cache_resource
def load_model():
    return joblib.load("mnist_model.pkl")

model = load_model()

st.title("📱 Klassificerare för Handskrivna Siffror (MNIST)")
st.write("Ladda upp en bild på en handskriven siffra (0-9) så gissar AI:n vilken siffra det är!")

uploaded_file = st.file_uploader("Välj en bildfil (jpg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Öppna bilden
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uppladdad bild")
        st.image(image, caption="Original", use_container_width=True)
        
    # PREPROCESSING (Det kritiska steget!)
    # 1. Konvertera till gråskala (L = luminance)
    img_gray = image.convert('L')
    
    # 2. Invertera färger om det är mörk siffra på ljust papper 
    # (MNIST har svart bakgrund och vit siffra. Om din bild är ljus med mörk text behöver den inverteras)
    # Vi kollar medelvärdet: om det är ljust i bakgrunden (> 127) inverterar vi.
    img_array_check = np.array(img_gray)
    if img_array_check.mean() > 127:
        img_gray = ImageOps.invert(img_gray)
        
    # 3. Ändra storlek till exakt 28x28 pixlar med högkvalitativ resampling
    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
    
    with col2:
        st.subheader("Efter Preprocessing")
        st.image(img_resized, caption="28x28 pixlar (inverterad/gråskala)", width=150)
        
    # 4. Gör om till numpy-array och skala till 0-1
    img_np = np.array(img_resized).reshape(1, -1) / 255.0
    
    # Prediktera
    prediction = model.predict(img_np)
    probabilities = model.predict_proba(img_np)
    confidence = np.max(probabilities) * 100
    
    st.markdown("---")
    st.success(f"### Modellen gissar att siffran är: **{prediction[0]}**")
    st.info(f"Förtroende (Confidence): **{confidence:.1f}%**")