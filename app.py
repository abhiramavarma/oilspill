import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import cv2

# --- App Configuration ---
st.set_page_config(
    page_title="Oil Spill Segmentation",
    page_icon="🌊",
    layout="wide"
)

# --- Constants ---
IMG_SIZE = (256, 256)
MODEL_PATH = 'best_model.pth' 
MIN_SPILL_AREA_PIXELS = 500


#  1. Define the U-Net Model Architecture 

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

#  2. Define the Preprocessing Transform

base_transform = A.Compose([
    A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])


#  3. Load the Model

@st.cache_resource
def load_pytorch_model():
    """Loads the PyTorch U-Net model and its weights."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNET(in_channels=3, out_channels=1).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        st.error(f"Please make sure the model file '{MODEL_PATH}' is in the same directory.")
        return None, None


#  4. The Prediction Function

def predict_and_analyze(model, device, image_bytes):
    """Preprocesses image, runs PyTorch model prediction, analyzes, and creates an overlay."""
    # 1. Preprocess
    pil_image = Image.open(image_bytes).convert('RGB')
    image_np = np.array(pil_image) # Original image as NumPy array
    
    transformed = base_transform(image=image_np)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # 2. Run Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        predicted_mask_tensor = (probs > 0.5).float()

    # 3. Analyze the mask and apply minimum area threshold
    spill_pixel_count = torch.sum(predicted_mask_tensor).item()
    if spill_pixel_count > MIN_SPILL_AREA_PIXELS:
        status = f"Oil Spill Detected ({int(spill_pixel_count)} spill pixels found)"
        status_color = "red"
        # Ensure mask is not cleared if spill is detected
        final_binary_mask_np = predicted_mask_tensor.squeeze().cpu().numpy()
    else:
        status = "No Spill Detected"
        status_color = "green"
        # Clear the mask if no significant spill
        final_binary_mask_np = np.zeros_like(predicted_mask_tensor.squeeze().cpu().numpy())

    # --- NEW STEP: Create the Overlay Image ---
    # Resize the original image to match the model's input/output size for overlay
    resized_original_image_for_overlay = cv2.resize(image_np, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Define the color for the oil spill (e.g., Red in BGR, then convert to RGB for matplotlib/PIL)
    # OpenCV uses BGR by default, so we'll define it as BGR and convert later if needed.
    OIL_SPILL_COLOR_BGR = (0, 0, 255) # Bright Red (B=0, G=0, R=255)
    ALPHA = 0.5 # Transparency factor (0.0 = fully transparent, 1.0 = fully opaque)

    # Create a colored overlay from the binary mask
    # We want a 3-channel image for color blending
    colored_spill_mask = np.zeros_like(resized_original_image_for_overlay, dtype=np.uint8)
    
    # Apply the color only where the mask is 1 (oil spill)
    # Note: final_binary_mask_np is (H, W), so we use a boolean mask to apply color
    colored_spill_mask[final_binary_mask_np == 1] = OIL_SPILL_COLOR_BGR # BGR here

    # Convert the resized original image to BGR for consistent blending with OpenCV
    resized_original_image_bgr = cv2.cvtColor(resized_original_image_for_overlay, cv2.COLOR_RGB2BGR)

    # Blend the original image with the colored spill mask
    # cv2.addWeighted works with BGR images
    overlay_image_bgr = cv2.addWeighted(resized_original_image_bgr, 1 - ALPHA, colored_spill_mask, ALPHA, 0)
    
    # Convert the final overlay image back to RGB for Streamlit/PIL display
    overlay_image_rgb = cv2.cvtColor(overlay_image_bgr, cv2.COLOR_BGR2RGB)
    
    # The original image that Streamlit displays should also be resized to match
    original_image_for_display = pil_image.resize(IMG_SIZE)

    return original_image_for_display, overlay_image_rgb, status, status_color



# --- Streamlit UI ---
st.title("🌊 Oil Spill Detection System (PyTorch)")
st.markdown("Upload a satellite image to segment and analyze for potential oil spills.")

model, device = load_pytorch_model()

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        with st.spinner('Analyzing the image...'):
            original_image, predicted_mask, status, status_color = predict_and_analyze(model, device, uploaded_file)

        st.markdown(f'<h2 style="color:{status_color}; text-align:center;">{status}</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption='Original Uploaded Image', use_container_width=True)
        with col2:
            st.image(predicted_mask, caption='Oil Spill Overlay', use_container_width=True)

        st.success("Analysis complete!")
else:
    st.warning("Model could not be loaded. The application cannot proceed.")
