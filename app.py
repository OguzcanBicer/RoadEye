import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Başlık
st.title("🛣️ RoadEye - Intelligent Pothole Detector (GPU Powered)")

# Cihaz belirleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.markdown(f"**Inference device:** `{device}`")

# Modeli yükle
@st.cache_resource
def load_model():
    model = torch.jit.load("roadeye_model.pt", map_location=device)
    model.to(device)
    model.eval()
    return model

model = load_model()

# Görüntü işleme adımları
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 için
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Görüntü yükleme arayüzü
uploaded_file = st.file_uploader("Bir yol görüntüsü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görüntü", width=400)

    # Görüntüyü modele hazırla
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Tahmin
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    # Sınıf etiketi
    label = "🕳️ Pothole Detected!" if pred_class == 1 else "✅ Road is Clear"

    # Sonuç göster
    st.subheader("🔎 Prediction")
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence:** `{confidence:.2%}`")
