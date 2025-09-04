import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import mahalanobis
from PIL import Image
import joblib
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import gdown
import os

# ================= Download helper =================
def download_file(file_id, output_path):
    """Download from Google Drive if file not already present"""
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# ================= Download all model/data files =================
download_file("1RHikBiZnXU1h8BrGzsVqkmUdexS4-yRx", "cancer_classification_model.h5")
download_file("13bEeboBHx0s7gPIFMC-B01i1mcxNpepz", "resnet50_feature_extractor.h5")
download_file("1LsVU7YdNKrWJVjjW360gVoC9FNRFQlh7", "features_array.npy")
download_file("1wVChR59-z55E9BhgAuZVaoA1TxLvpcyW", "mean_vector.npy")
download_file("1Z_IdVqyjKjqxb-qdQkhF4SvZsS8vpES2", "cov_matrix.pkl")
download_file("19-chWleyeNDcmGrUzIdG0CA8N9XXdKn9", "ood_threshold.pkl")
download_file("12yBIgkVf8rBq3xbGn_MU5wqpmOs9XGGJ", "cancer_type_model.pkl")
download_file("1HTx6VbFQwhdEtob1N44MWSvRIVjw_zol", "severity_stage_model.pkl")
download_file("1qelYWyJ1im0wGrbn7pbtC8tS_iZYiCc3", "bcss_unet_model.pth")

# ================= Load TensorFlow Models =================
cnn_model = load_model("cancer_classification_model.h5")
feature_extractor = load_model("resnet50_feature_extractor.h5")

# ================= Load OOD Detection Data =================
features_array = np.load("features_array.npy")
mean_vector = np.load("mean_vector.npy")
with open("cov_matrix.pkl", "rb") as f:
    cov_matrix = pickle.load(f)
with open("ood_threshold.pkl", "rb") as f:
    ood_threshold = pickle.load(f)

# ================= Load Severity Models =================
type_model = joblib.load("cancer_type_model.pkl")
stage_model = joblib.load("severity_stage_model.pkl")

# ================= Feature Extractor for Severity =================
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
severity_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# ================= Label Maps =================
cancer_classes = {
    0: "Breast Cancer (Cancerous)",
    1: "Breast Cancer (Non-Cancerous)",
    2: "Lung Cancer (Cancerous)",
    3: "Lung Cancer (Non-Cancerous)",
    4: "Skin Cancer (Cancerous)",
    5: "Skin Cancer (Non-Cancerous)",
    6: "Brain Cancer (Cancerous)",
    7: "Brain Cancer (Non-Cancerous)",
    8: "Oral Cancer (Cancerous)",
    9: "Oral Cancer (Non-Cancerous)"
}
ood_label = "Not Defined (Out-of-Distribution)"
cancer_types = ["Papillary Carcinoma", "Mucinous Carcinoma", "Lobular Carcinoma", "Ductal Carcinoma"]
severity_stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]

# ================= UNet Segmentation Model =================
class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconv4 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.final = torch.nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(torch.nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(torch.nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(torch.nn.MaxPool2d(2)(enc3))
        bottleneck = self.bottleneck(torch.nn.MaxPool2d(2)(enc4))
        dec4 = self.dec4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1), dim=1))
        return self.final(dec1)

# Load UNet Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load("bcss_unet_model.pth", map_location=device))
unet_model.eval()

seg_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ================= Streamlit App =================
st.title("AI-Powered Cancer Diagnosis: Classification, Segmentation & Severity Prediction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def extract_features(img_pil):
    img_pil = img_pil.convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img_pil), axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    return features.flatten()

def classify_image(img_pil):
    img_rgb = img_pil.convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img_rgb) / 255.0, axis=0)
    predictions = cnn_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    test_features = extract_features(img_pil)
    distance = mahalanobis(test_features, mean_vector, cov_matrix.precision_)
    if distance > ood_threshold:
        return ood_label, None, distance
    else:
        return cancer_classes[predicted_class], predicted_class, distance

def segment_breast_cancer(img_pil):
    img_tensor = seg_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = unet_model(img_tensor)
        mask = torch.sigmoid(output).squeeze(0).cpu().numpy()
    return mask[0]

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file)
    st.image(img_pil, caption="Uploaded Image", use_column_width=True)
    label, pred_index, distance = classify_image(img_pil)
    st.markdown(f"### Prediction: `{label}`")
    if label != ood_label:
        st.markdown(f"**Mahalanobis Distance:** `{distance:.2f}`")

    if label == "Breast Cancer (Cancerous)":
        st.markdown("---")
        st.subheader("Segmented Tumor Region")
        segmented_mask = segment_breast_cancer(img_pil)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(img_pil)
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(segmented_mask, cmap="gray")
        ax[1].set_title("Predicted Mask")
        ax[1].axis("off")
        st.pyplot(fig)

        # Overlay Mask
        st.markdown("### Visual Overlay for Validation")
        img_resized = img_pil.resize((128, 128)).convert("RGB")
        fig_overlay, ax_overlay = plt.subplots()
        ax_overlay.imshow(img_resized)
        ax_overlay.imshow(segmented_mask, cmap="Reds", alpha=0.4)
        ax_overlay.axis("off")
        ax_overlay.set_title("Overlay of Prediction on Original Image")
        st.pyplot(fig_overlay)

        # Severity Prediction
        st.markdown("---")
        st.subheader("Severity Classification")

        resized_img = img_pil.convert("RGB").resize((224, 224))
        img_array = image.img_to_array(resized_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        feature = severity_model.predict(img_array)
        predicted_type = type_model.predict(feature)
        predicted_stage = stage_model.predict(feature)

        st.markdown(f"**Predicted Cancer Type:** `{cancer_types[predicted_type[0]]}`")
        st.markdown(f"**Predicted Severity Stage:** `{severity_stages[predicted_stage[0]]}`")
