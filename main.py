import os, cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm


# --- Dataset class ---
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, aug=None):
        self.names = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.img_dir, self.mask_dir, self.aug = img_dir, mask_dir, aug

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = cv2.imread(os.path.join(self.img_dir, name))[:, :, ::-1]
        mask = cv2.imread(os.path.join(self.mask_dir, os.path.splitext(name)[0] + ".png"), cv2.IMREAD_GRAYSCALE)

        if self.aug:
            out = self.aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        img = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
        mask = mask.astype(np.int64)
        return torch.from_numpy(img), torch.from_numpy(mask)


def main():
    # --- Paths ---
    train_imgs = "dataset_seg/images/train"
    train_masks = "dataset_seg/masks/train"
    val_imgs   = "dataset_seg/images/val"
    val_masks  = "dataset_seg/masks/val"

    # --- Augmentations ---
    train_tf = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(0.5),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
    ])
    val_tf = A.Compose([A.Resize(512, 512)])

    # --- DataLoaders ---
    train_dl = DataLoader(
        SegDataset(train_imgs, train_masks, train_tf),
        batch_size=2, shuffle=True, num_workers=2, pin_memory=True
    )
    val_dl = DataLoader(
        SegDataset(val_imgs, val_masks, val_tf),
        batch_size=2, shuffle=False, num_workers=2, pin_memory=True
    )

    # --- Model ---
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b1",  # dùng EfficientNet-B1
        encoder_weights="imagenet",
        classes=6,          # 6 lớp mask
        activation=None
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Training on {device.upper()} | Encoder: EfficientNet-B1 | Image size: 512×512")

    # --- Optimizer & Loss ---
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss(mode='multiclass')

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    best = 0.0
    epochs = 60

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}", unit="batch")

        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()

            # --- Mixed precision training ---
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = 0.5 * ce(out, y) + 0.5 * dice(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation ---
        model.eval(); scores = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                scores.append(1.0 - dice(preds, y).item())

        val_dice = np.mean(scores)
        print(f"✅ Epoch {ep+1}/{epochs} | Train Loss: {total_loss/len(train_dl):.4f} | Val Dice: {val_dice:.4f}")

        # --- Save best model ---
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), "best_unetpp_effb1.pth")
            print(f"Saved new best model with Dice {best:.4f}")

        torch.cuda.empty_cache()  # dọn VRAM mỗi epoch

    print(f"Training complete! Best Dice = {best:.4f}")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
