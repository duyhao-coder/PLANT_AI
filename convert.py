import os, cv2, numpy as np, glob

# === BẢNG MÀU (RGB) → ID (theo bảng bạn đã chốt) ===
COLOR2ID = {
    (0, 0, 0): 0,         # background
    (128, 0, 0): 1,       # healthy
    (128, 128, 0): 2,     # Alternaria
    (128, 0, 128): 3,     # Brown
    (0, 0, 128): 4,       # Gray
    (0, 128, 0): 5,       # Rust
}

# === THƯ MỤC MASK ===
MASK_DIRS = [
    "dataset_seg/masks/train",
    "dataset_seg/masks/val",
]

def convert_mask(mask_bgr):
    """Chuyển mask màu BGR -> mask ID"""
    mask = mask_bgr[:, :, ::-1]  # BGR → RGB
    h, w = mask.shape[:2]
    id_mask = np.zeros((h, w), np.uint8)
    for rgb, cid in COLOR2ID.items():
        match = np.all(mask == np.array(rgb, dtype=np.uint8), axis=2)
        id_mask[match] = cid
    return id_mask

for folder in MASK_DIRS:
    print(f"🔄 Đang xử lý {folder} ...")
    masks = glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg"))
    for path in masks:
        mask = cv2.imread(path)
        if mask is None:
            print("⚠️ Không đọc được:", path)
            continue
        id_mask = convert_mask(mask)
        cv2.imwrite(path, id_mask)  # Ghi đè
print("✅ Đã convert toàn bộ mask sang ID (0–5)!")
