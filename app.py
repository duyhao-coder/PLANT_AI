import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# --- Cấu hình ---
MODEL_PATH = "best_unetpp_effb1.pth"
NUM_CLASSES = 6
IMG_SIZE = 512

# --- Load model ---
@st.cache_resource
def load_segmentation_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b1",
        encoder_weights=None,
        classes=NUM_CLASSES,
        activation=None
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

model, device = load_segmentation_model()
st.sidebar.success(f"🚀 Model loaded on {device.upper()}")

# --- Bảng ánh xạ ID -> Tên bệnh ---
id2name = {
    0: "Background",
    1: "Healthy",
    2: "Alternaria leaf spot",
    3: "Brown spot",
    4: "Gray spot",
    5: "Rust"
}

# --- Title ---
st.title("🍃 Plant Disease Segmentation App")
st.markdown("Tải ảnh lá cây lên để phát hiện bệnh.")

# --- Upload image ---
uploaded_file = st.file_uploader("Chọn ảnh lá cây...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Đọc file ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)[:, :, ::-1]
    orig = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Hiển thị ảnh gốc
    st.image(orig, caption="Ảnh gốc", use_container_width=True)

    if st.button("🔍 Predict"):
        # Chuẩn bị tensor
        x = np.transpose(orig.astype(np.float32) / 255.0, (2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0).to(device)

        # Dự đoán
        with torch.no_grad():
            pred = model(x)
            mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # --- Lọc nhiễu nhỏ ---
        counts = np.bincount(mask.flatten(), minlength=NUM_CLASSES)
        min_pixels = mask.size * 0.005
        valid_classes = [i for i, c in enumerate(counts) if c > min_pixels]
        diseases = [id2name[c] for c in valid_classes if c not in (0, 1)]

        if diseases:
            st.success("🩺 Phát hiện bệnh: **" + ", ".join(diseases) + "**")
        else:
            st.info("🍃 Lá khỏe mạnh, không phát hiện bệnh.")

        # --- Tạo overlay ---
        COLORS = np.array([
            [0, 0, 0],
            [128, 0, 0],
            [128, 128, 0],
            [128, 0, 128],
            [0, 0, 128],
            [0, 128, 0],
        ], dtype=np.uint8)

        mask_color = COLORS[mask]
        overlay = cv2.addWeighted(orig, 0.7, mask_color, 0.3, 0)

        # --- Vẽ contour ---
        overlay_contour = overlay.copy()
        for cls_id in range(2, NUM_CLASSES):  # bỏ background & healthy
            mask_cls = (mask == cls_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (0, 255, 0)
            overlay_contour = cv2.drawContours(overlay_contour, contours, -1, color, 2)

        # --- Hiển thị 3 ảnh ---
        st.subheader("🔎 Kết quả dự đoán")
        col1, col2, col3 = st.columns(3)
        col1.image(orig, caption="Original", use_container_width=True)
        col2.image(mask_color, caption="Predicted Mask", use_container_width=True)
        col3.image(overlay_contour, caption="Overlay Polygon", use_container_width=True)

        # --- Lưu ảnh kết quả ---
        cv2.imwrite("pred_overlay.png", cv2.cvtColor(overlay_contour, cv2.COLOR_RGB2BGR))
        st.download_button(
            label="💾 Tải ảnh kết quả",
            data=open("pred_overlay.png", "rb").read(),
            file_name="pred_overlay.png",
            mime="image/png"
        )
