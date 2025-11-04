import matplotlib.pyplot as plt

# 📊 Dữ liệu copy thủ công từ log (ví dụ từ epoch 1–27)
train_losses = [0.3494, 0.1426, 0.0956, 0.0725, 0.0607, 0.0570, 0.0549, 0.0507, 
                0.0528, 0.0489, 0.0493, 0.0464, 0.0454, 0.0450, 0.0428, 0.0424, 
                0.0435, 0.0436, 0.0416, 0.0404, 0.0410, 0.0394, 0.0401, 0.0395,
                0.0392, 0.0402, 0.0389]

val_dices = [0.8804, 0.9076, 0.9277, 0.9536, 0.9533, 0.9456, 0.9563, 0.9544,
             0.9577, 0.9577, 0.9477, 0.9610, 0.9576, 0.9562, 0.9531, 0.9601,
             0.9478, 0.9515, 0.9576, 0.9584, 0.9550, 0.9599, 0.9588, 0.9585,
             0.9550, 0.9501, 0.9540]

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss", color="tab:red")
plt.plot(val_dices, label="Val Dice", color="tab:blue")
plt.xlabel("Epoch")
plt.ylabel("Score / Loss")
plt.title("Training Curve (U-Net++ EfficientNet-B1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_curve_from_log.png")
plt.show()
