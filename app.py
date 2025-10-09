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

# Fix matplotlib cache directory issue
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_cache')

import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(
    page_title="U-Net Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

# --- Global Settings ---
DEVICE = torch.device("cpu")
MODEL_PATH = 'best_model.pth' # From Code 2
IMG_SIZE = (256, 256)      # Model's expected input size from Code 2
DEFAULT_THRESHOLD = 0.5    # Standard starting threshold for sigmoid

# --- 1. U-Net Model Architecture (from Code 2) ---
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

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)

# --- 2. Preprocessing Transform (from Code 2) ---
base_transform = A.Compose([
    A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# --- 3. Model Loader (for U-Net from Code 2) ---
@st.cache_resource
def load_unet_model():
    """
    Loads the custom PyTorch U-Net model and its weights.
    Cached by Streamlit to load only once.
    """
    try:
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        # Check if model file exists before loading
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

# --- 4. Inference, Visualization, and Analysis (Style from Code 1) ---
@torch.no_grad()
def predict_and_visualize(model, image, threshold):
    """
    Runs inference using the U-Net model and generates the styled 4-panel plot and stats.
    """
    try:
        # Preserve original for sizing and final display
        original = np.array(image.convert("RGB"))
        oh, ow = original.shape[:2]

        # Preprocess using the U-Net's specific transform
        transformed = base_transform(image=original)
        x = transformed['image'].unsqueeze(0).to(DEVICE)

        # Get model prediction
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy() # (256, 256)

        # IMPORTANT: Resize probability map back to original image size for accurate overlay
        prob_resized = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = (prob_resized >= float(threshold)).astype(np.uint8)

        # Create overlay (style from Code 1)
        orig_gray_for_display = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        overlay_rgb = cv2.cvtColor(orig_gray_for_display, cv2.COLOR_GRAY2RGB)
        overlay_colored = overlay_rgb.copy()
        overlay_colored[mask == 1] = [255, 0, 0] # Red for spill
        blended = cv2.addWeighted(overlay_rgb, 0.7, overlay_colored, 0.3, 0)

        # Calculate statistics (style from Code 1)
        total_px = mask.size
        oil_px = int(mask.sum())
        oil_pct = 100.0 * oil_px / max(total_px, 1)
        conf_max = float(prob_resized.max())
        conf_mean_spill = float(prob_resized[mask == 1].mean()) if oil_px > 0 else 0.0

        # --- Generate 4-Panel Plot (style from Code 1) ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='#0E1117')
        fig.suptitle("U-Net Spill Analysis Results", fontsize=18, fontweight="bold", color='white')

        axes[0, 0].imshow(orig_gray_for_display, cmap="gray")
        axes[0, 0].set_title(f"Input Image ({ow}×{oh})", color='white')
        axes[0, 0].axis("off")

        im1 = axes[0, 1].imshow(prob_resized, cmap="hot", vmin=0, vmax=1)
        axes[0, 1].set_title("Oil Spill Probability", color='white')
        axes[0, 1].axis("off")
        cbar = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        axes[1, 0].imshow(mask, cmap="Reds", vmin=0, vmax=1)
        axes[1, 0].set_title(f"Detection (Threshold: {float(threshold):.2f})", color='white')
        axes[1, 0].axis("off")

        axes[1, 1].imshow(blended)
        axes[1, 1].set_title("Oil Spill Overlay", color='white')
        axes[1, 1].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)
        result_image = Image.open(buf)

        # --- Generate Statistics Markdown (style from Code 1) ---
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
        | **Threshold Used** | {float(threshold):.2f} |
        | **Image Size** | {ow} × {oh} pixels |
        """
        return result_image, results_text

    except Exception as e:
        st.error(f"❌ Error during analysis: {e}")
        return None, None

# --- 5. Streamlit UI - Instructions First Layout ---
st.title("🌊 Oil Spill Detection System")
st.markdown("AI-powered oil spill detection from satellite imagery.")

# Load Model
model = load_unet_model()

if model:
    # Create two columns: Left = Instructions + Browse, Right = Results
    left_col, right_col = st.columns([1, 1])

    # LEFT COLUMN: Browse first, then Instructions
    with left_col:
        # Browse/Upload in first place
        st.markdown("### 📤 Browse Files")
        uploaded_file = st.file_uploader(
            "Choose a satellite image file",
            type=["jpg", "png", "jpeg"],
            help="Select a satellite or aerial image for oil spill analysis"
        )

        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)

            st.markdown("**Detection Settings:**")
            threshold = st.slider(
                "Sensitivity Level",
                min_value=0.1, max_value=0.9,
                value=DEFAULT_THRESHOLD, step=0.05,
                help="Adjust how sensitive the detection should be"
            )

            if st.button("🔍 Detect Oil Spills", type="primary"):
                with st.spinner('🤖 Analyzing image with AI model...'):
                    result_plot, result_stats = predict_and_visualize(model, input_image, threshold)

                if result_plot and result_stats:
                    # Store results for right column display
                    st.session_state.analysis_complete = True
                    st.session_state.result_plot = result_plot
                    st.session_state.result_stats = result_stats
                    st.session_state.input_image = input_image
                    st.rerun()
                else:
                    st.error("❌ Could not analyze the image. Please try another file.")

        st.markdown("---")

        # Instructions in second place
        st.markdown("### 📋 Instructions")
        st.markdown("""
        **How to detect oil spills:**

        1. **Browse** and select a satellite image file
        2. **Adjust** the detection sensitivity if needed
        3. **Click** 'Detect Oil Spills' to analyze
        4. **View results** in the right panel

        **Sensitivity Guide:**
        - 🔴 **Low (0.1-0.3)**: High sensitivity, may detect small spills
        - 🟡 **Medium (0.4-0.6)**: Balanced detection
        - 🟢 **High (0.7-0.9)**: Low sensitivity, fewer false positives

        **Tips:**
        - Use higher sensitivity for suspicious areas
        - Lower sensitivity reduces false alarms
        - Analysis takes a few seconds
        """)

    # RIGHT COLUMN: Results Display
    with right_col:
        st.markdown("### 📊 Analysis Results")

        if uploaded_file is not None and 'analysis_complete' in st.session_state:
            # Show input image
            st.markdown("**Input Image:**")
            st.image(st.session_state.input_image, width=350)

            # Show detection result
            oil_pct = float(st.session_state.result_stats.split("Oil Coverage")[1].split("%")[0].split()[-1])
            if oil_pct > 1:
                st.success(f"🚨 **OIL SPILL DETECTED** ({oil_pct:.1f}% of image)")
            else:
                st.success(f"✅ **NO SPILL DETECTED** ({oil_pct:.1f}% of image)")

            # Show analysis visualization
            st.image(st.session_state.result_plot, caption="AI Analysis Results", width=500)

            # Show detailed statistics
            st.markdown("**Detailed Statistics:**")
            st.markdown(st.session_state.result_stats)

        elif uploaded_file is not None:
            st.markdown("**Input Image Preview:**")
            input_image = Image.open(uploaded_file)
            st.image(input_image, width=350, caption="Image loaded - Click 'Detect Oil Spills' to analyze")
            st.info("👈 Use the controls in the left panel to analyze this image.")

        else:
            st.info("📤 Upload an image in the left panel and run analysis to see results here.")

else:
    st.error("❌ AI model could not be loaded. Please ensure 'best_model.pth' exists.")
