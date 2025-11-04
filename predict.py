import torch, cv2, numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# --- Config ---
MODEL_PATH = "best_unetpp_effb1.pth"
IMG_PATH = "001171.jpg"
NUM_CLASSES = 6         # số lớp mask (0–5)
IMG_SIZE = 512

# --- Load model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.UnetPlusPlus(
    encoder_name="timm-efficientnet-b1",
    encoder_weights=None,   # inference ko cần pretrained
    classes=NUM_CLASSES,
    activation=None
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"🚀 Loaded model from {MODEL_PATH} on {device.upper()}")

# --- Load and preprocess image ---
img = cv2.imread(IMG_PATH)[:, :, ::-1]
orig = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
x = np.transpose(orig.astype(np.float32) / 255.0, (2, 0, 1))
x = torch.from_numpy(x).unsqueeze(0).to(device)

# --- Predict ---
with torch.no_grad():
    pred = model(x)
    mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)

# --- Bảng ánh xạ ID -> tên bệnh ---
id2name = {
    0: "Background",
    1: "Healthy",
    2: "Alternaria leaf spot",
    3: "Brown spot",
    4: "Gray spot",
    5: "Rust"
}

# --- 🧩 Lọc nhiễu nhỏ (noise) ---
counts = np.bincount(mask.flatten(), minlength=NUM_CLASSES)
min_pixels = mask.size * 0.005  # loại bỏ class chiếm <0.5% diện tích ảnh
valid_classes = [i for i, c in enumerate(counts) if c > min_pixels]

# --- Xác định bệnh thật sự ---
diseases = [id2name[c] for c in valid_classes if c not in (0, 1)]

if diseases:
    print("🩺 Detected disease(s):", ", ".join(diseases))
else:
    print("🍃 No disease detected (Healthy leaf)")

# --- Tạo overlay ---
COLORS = np.array([
    [0, 0, 0],        # 0 - Background
    [128, 0, 0],      # 1 - Healthy
    [128, 128, 0],    # 2 - Alternaria leaf spot
    [128, 0, 128],    # 3 - Brown spot
    [0, 0, 128],      # 4 - Gray spot
    [0, 128, 0],      # 5 - Rust
], dtype=np.uint8)

mask_color = COLORS[mask]
overlay = cv2.addWeighted(orig, 0.7, mask_color, 0.3, 0)

# --- Hiển thị ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(orig)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_color)
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
for cls_id in range(2, NUM_CLASSES):  # bỏ background & healthy
    mask_cls = (mask == cls_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = (0, 255, 0)
    overlay = cv2.drawContours(overlay.copy(), contours, -1, color, 2)
plt.imshow(overlay)
plt.title("Overlay with Polygon Contours")
plt.axis("off")

plt.tight_layout()
plt.show()


# --- Lưu kết quả ---
cv2.imwrite("pred_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print("Saved overlay as pred_overlay.png")


