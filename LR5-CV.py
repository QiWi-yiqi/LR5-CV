import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# -------------------------------
# Step 1: Streamlit page setup
# -------------------------------
st.set_page_config(
    page_title="VisionNet-CPU",
    layout="centered"
)

st.title("🦖VisionNet-CPU: Web-Based Image Classification Using Pretrained ResNet18")
st.write("This application performs image recognition using a pretrained ResNet18 model on CPU.")

# -------------------------------
# Step 3: Force CPU device
# -------------------------------
device = torch.device("cpu")
st.write(f"Running on device: {device}")

# -------------------------------
# Step 4: Load pretrained ResNet18
# -------------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()
model.to(device)

# -------------------------------
# Step 5: Image preprocessing
# -------------------------------
preprocess = weights.transforms()

# Class labels
class_names = weights.meta["categories"]

# -------------------------------
# Step 6: Upload image interface
# -------------------------------
uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Step 7: Convert image and inference
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing image...")

    img_tensor = preprocess(image).unsqueeze(0).to(device)

    # No gradient computation
    with torch.no_grad():
        output = model(img_tensor)

    # Step 8: Softmax + Top-5 predictions
    probabilities = F.softmax(output[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    st.subheader("Top-5 Predictions")
    results = []

    for i in range(5):
        results.append([class_names[top5_catid[i]], float(top5_prob[i])])

    df = pd.DataFrame(results, columns=["Class", "Probability"])
    st.table(df)

    # Step 9: Bar chart
    st.subheader("Prediction Probabilities")
    st.bar_chart(df.set_index("Class"))

# Step 10: Discussion Section
st.subheader("Discussion of Results")
st.write("""
The system classifies uploaded images into ImageNet object categories using ResNet18.
The model follows the path: image upload → preprocessing → tensor conversion → model inference → softmax → top-5 prediction output.
The accuracy depends on image clarity, lighting, object visibility, and similarity to ImageNet training data.
""")

