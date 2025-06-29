import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Page setup
st.set_page_config(
    page_title="RoadEye - Multi-Model Pothole Analysis",
    layout="wide"
)

# Title
st.title("🛣️ RoadEye - Multi-Model Pothole Analysis")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation (for CNNs and ViT)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Model loader
@st.cache_resource
def load_models():
    return {
        "InceptionV3": torch.jit.load("models/inception.pt", map_location=device).eval(),
        "MobileNetV2": torch.jit.load("models/mobilenet.pt", map_location=device).eval(),
        "Vision Transformer": torch.jit.load("models/vit.pt", map_location=device).eval(),
        "YOLOv11": YOLO("models/yolov11.pt")
    }

models = load_models()

# Layout: 3/10 - 2/10 - 5/10
col1, col2, col3 = st.columns([4, 3, 4])

# Left column: image upload & preview
with col1:
    uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=225)
        input_tensor = transform(image).unsqueeze(0).to(device)
        st.markdown("##### Image successfully loaded.")
    else:
        st.stop()

# Middle column: classification model results
with col2:
    st.markdown("### 🔍 Predictions")
    for name in ["InceptionV3", "MobileNetV2", "Vision Transformer"]:
        with torch.no_grad():
            output = models[name](input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = torch.max(probs).item()
            label = "🕳️ Pothole" if pred == 1 else "✅ Clear Road"
            st.markdown(f"**{name}**")
            st.write(f"- {label} ({conf:.2%})")

# Right column: YOLO visualization
with col3:
    st.markdown("### 📦 YOLOv11 Detection")
    yolo_results = models["YOLOv11"](image)
    yolo_results[0].save(filename="yolo_result.jpg")
    st.image("yolo_result.jpg", caption="YOLOv11 Output", width=450)
