import os
import io
import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import streamlit as st
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set matplotlib cache directory to avoid import issues
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_cache')




import matplotlib.pyplot as plt

# App configuration
st.set_page_config(
    page_title="U-Net Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

# Model settings
DEVICE = torch.device("cpu")
MODEL_PATH = 'best_model.pth'
IMG_SIZE = (256, 256)
DEFAULT_THRESHOLD = 0.5

# U-Net building block with two conv layers
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

# Main U-Net model with encoder-decoder architecture
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder: upsampling path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path with skip connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# Image preprocessing for the model
base_transform = A.Compose([
    A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# Load the trained U-Net model
@st.cache_resource
def load_unet_model():
    try:
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at '{MODEL_PATH}'. Please place the model in the correct directory.")
            return None
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"✅ U-Net model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

# Run inference and create visualization
@torch.no_grad()
def predict_and_visualize(model, image, threshold):
    try:
        # Convert image and get dimensions
        original = np.array(image.convert("RGB"))
        oh, ow = original.shape[:2]

        # Preprocess and run model
        transformed = base_transform(image=original)
        x = transformed['image'].unsqueeze(0).to(DEVICE)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # Resize prediction to match original image size
        prob_resized = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = (prob_resized >= float(threshold)).astype(np.uint8)

        # Create visualization overlay
        gray_image = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        overlay = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        overlay[mask == 1] = [255, 0, 0]  # Red overlay for detected spills
        blended = cv2.addWeighted(overlay, 0.7, cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB), 0.3, 0)

        # Calculate spill statistics
        total_px = mask.size
        oil_px = int(mask.sum())
        oil_pct = 100.0 * oil_px / max(total_px, 1)
        conf_max = float(prob_resized.max())
        conf_mean_spill = float(prob_resized[mask == 1].mean()) if oil_px > 0 else 0.0

        # Create 4-panel visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Oil Spill Analysis Results", fontsize=16, fontweight="bold")

        # Input image
        axes[0, 0].imshow(gray_image, cmap="gray")
        axes[0, 0].set_title(f"Input Image ({ow}×{oh})")
        axes[0, 0].axis("off")

        # Probability heatmap
        im1 = axes[0, 1].imshow(prob_resized, cmap="hot", vmin=0, vmax=1)
        axes[0, 1].set_title("Spill Probability")
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1])

        # Detection mask
        axes[1, 0].imshow(mask, cmap="Reds", vmin=0, vmax=1)
        axes[1, 0].set_title(f"Detection (Threshold: {threshold:.2f})")
        axes[1, 0].axis("off")

        # Overlay visualization
        axes[1, 1].imshow(blended)
        axes[1, 1].set_title("Spill Overlay")
        axes[1, 1].axis("off")

        plt.tight_layout()

        # Save plot to memory
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        result_image = Image.open(buf)

        # Generate statistics summary
        severity = 'HIGH' if oil_pct > 10 else 'MODERATE' if oil_pct > 5 else 'LOW' if oil_pct > 1 else 'MINIMAL'
        status_icon = '🚨' if oil_pct > 1 else '✅'
        status_text = 'OIL SPILL DETECTED' if oil_pct > 1 else 'No significant oil detected'

        results_text = f"""
        | Metric | Value |
        | :--- | :--- |
        | **Status** | **{status_icon} {status_text}** |
        | **Severity** | {severity} |
        | **Oil Coverage** | {oil_pct:.2f}% of image |
        | **Oil Pixels** | {oil_px:,} / {total_px:,} |
        | **Max Confidence** | {conf_max:.3f} |
        | **Mean Spill Confidence** | {conf_mean_spill:.3f} |
        | **Threshold Used** | {threshold:.2f} |
        | **Image Size** | {ow} × {oh} pixels |
        """

        return result_image, results_text

    except Exception as e:
        st.error(f"❌ Error during analysis: {e}")
        return None, None

# Main application interface
st.title("🌊 Oil Spill Detection System")
st.markdown("AI-powered oil spill detection from satellite imagery.")

model = load_unet_model()

if model:
    # Two-column layout
    left_col, right_col = st.columns([1, 1])

    # Left column: file upload and controls
    with left_col:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a satellite image",
            type=["jpg", "png", "jpeg"]
        )

        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)

            # Clear old results when new file is selected
            if 'previous_file' not in st.session_state or st.session_state.previous_file != uploaded_file.name:
                st.session_state.previous_file = uploaded_file.name
                # Clear previous analysis
                keys_to_delete = ['analysis_complete', 'result_plot', 'result_stats', 'input_image']
                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

            threshold = st.slider(
                "Detection Sensitivity",
                min_value=0.1, max_value=0.9,
                value=DEFAULT_THRESHOLD, step=0.05
            )

            if st.button("🔍 Detect Oil Spills", type="primary"):
                with st.spinner('Analyzing image...'):
                    result_plot, result_stats = predict_and_visualize(model, input_image, threshold)

                if result_plot and result_stats:
                    # Store results in session state
                    st.session_state.analysis_complete = True
                    st.session_state.result_plot = result_plot
                    st.session_state.result_stats = result_stats
                    st.session_state.input_image = input_image
                    st.rerun()

        st.markdown("---")
        st.markdown("### 📋 How to Use")
        st.markdown("""
        **Steps:**
        1. Upload a satellite image
        2. Adjust sensitivity if needed
        3. Click 'Detect Oil Spills'

        **Sensitivity Guide:**
        - 🔴 **Low (0.1-0.3)**: High sensitivity
        - 🟡 **Medium (0.4-0.6)**: Balanced
        - 🟢 **High (0.7-0.9)**: Low sensitivity
        """)

    # Right column: results and image preview
    with right_col:
        st.markdown("### 📊 Results")

        if uploaded_file is not None:
            # Show current image immediately
            input_image = Image.open(uploaded_file)
            st.image(input_image, width=350, caption="Current image")

            # Show analysis results if available
            if 'analysis_complete' in st.session_state:
                oil_pct = float(st.session_state.result_stats.split("Oil Coverage")[1].split("%")[0].split()[-1])
                if oil_pct > 1:
                    st.error("🚨 **OIL SPILL DETECTED** ")
                else:
                    st.success("✅ **NO SPILL DETECTED** ")

                st.image(st.session_state.result_plot, caption="Analysis Results", width=500)
                st.markdown("**Statistics:**")
                st.markdown(st.session_state.result_stats)
            else:
                st.info("👈 Click 'Detect Oil Spills' to analyze this image.")
        else:
            st.info("Upload an image to see results here.")

else:
    st.error("❌ Model not loaded. Check that 'best_model.pth' exists.")
