"""
Fine-tuning SegFormer-B0 on RUGD Off-Road Semantic Segmentation Dataset.
Trains on NVIDIA RTX 2050 GPU using PyTorch Mixed Precision (AMP),
computes mIoU & Pixel Accuracy, and plots loss curves for project evaluation.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Class Definitions
NUM_CLASSES = 7
CLASS_NAMES = [
    "Background",
    "Dirt / Path",
    "Grass",
    "Trees / Veg",
    "Rocks / Rough",
    "Water",
    "Sky"
]

class RUGDSegmentationDataset(Dataset):
    """Custom PyTorch Dataset for RUGD off-road segmentation pairs."""

    def __init__(self, data_dir, image_files, processor, target_size=(512, 512)):
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.image_files = image_files
        self.processor = processor
        self.target_size = target_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        base_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, f"{base_name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{base_name}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Resize to 512x512
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        # Process image via SegFormer processor
        encoded = self.processor(image, return_tensors="pt")
        pixel_values = encoded.pixel_values.squeeze(0)  # (3, 512, 512)

        # Mask as LongTensor
        labels = torch.from_numpy(np.array(mask, dtype=np.int64))  # (512, 512)

        return {"pixel_values": pixel_values, "labels": labels}

def compute_miou(preds, targets, num_classes=NUM_CLASSES, ignore_index=0):
    """Computes Mean Intersection over Union (mIoU) across semantic classes."""
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    valid_ious = [v for v in ious if not np.isnan(v)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    return miou, ious

def train_model():
    print("=" * 60)
    print("  AUTONAV AI: SEGFORMER-B0 RUGD FINE-TUNING PIPELINE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute hardware: {device}")
    if torch.cuda.is_available():
        print(f"[GPU Model] {torch.cuda.get_device_name(0)} (VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB)")

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "datasets", "RUGD_train")
    model_dir = os.path.join(base_dir, "models", "segformer-b0")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Dataset Indexing & Splitting (80% Train, 20% Val)
    all_files = sorted([f.replace(".jpg", "") for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(".jpg")])
    np.random.seed(42)
    np.random.shuffle(all_files)

    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"[Dataset] Total samples: {len(all_files)} | Train: {len(train_files)} | Validation: {len(val_files)}")

    # 2. Processor & Model Setup
    processor = SegformerImageProcessor.from_pretrained(model_dir)
    print(f"[Model] Initializing SegFormer-B0 with custom 7-class off-road classification head...")
    
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # DataLoaders
    train_dataset = RUGDSegmentationDataset(data_dir, train_files, processor)
    val_dataset = RUGDSegmentationDataset(data_dir, val_files, processor)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 3. Optimizer & Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.cuda.amp.GradScaler()

    # 4. Training Loop
    EPOCHS = 5
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_pixel_acc": []}

    print("\n[Training] Commencing Fine-Tuning Run (5 Epochs with Mixed Precision)...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits  # (B, num_classes, H/4, W/4)

                # Upsample logits to label resolution (512x512)
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                loss = criterion(upsampled_logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)

        # Validation Step
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=pixel_values)
                    logits = outputs.logits
                    upsampled_logits = nn.functional.interpolate(
                        logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    v_loss = criterion(upsampled_logits, labels)

                running_val_loss += v_loss.item()
                preds = upsampled_logits.argmax(dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        epoch_val_loss = running_val_loss / len(val_loader)
        cat_preds = torch.cat(all_preds, dim=0)
        cat_labels = torch.cat(all_labels, dim=0)

        # Accuracy & mIoU
        valid_mask = (cat_labels != 0)
        pixel_acc = float((cat_preds[valid_mask] == cat_labels[valid_mask]).sum()) / max(1, int(valid_mask.sum())) * 100
        val_miou, per_class_ious = compute_miou(cat_preds, cat_labels)
        val_miou_pct = val_miou * 100

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_miou"].append(round(val_miou_pct, 2))
        history["val_pixel_acc"].append(round(pixel_acc, 2))

        epoch_dur = round(time.time() - epoch_start, 1)
        print(f"Epoch [{epoch}/{EPOCHS}] ({epoch_dur}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Pixel Acc: {pixel_acc:.1f}% | "
              f"mIoU: {val_miou_pct:.1f}%")

    total_time = round(time.time() - start_time, 1)
    print(f"\n[Training] Completed 5 epochs in {total_time} seconds (~{round(total_time/60, 1)} min)!")

    # 5. Save Fine-Tuned Weights
    save_model_path = os.path.join(base_dir, "models", "segformer_b0_rugd_finetuned.pt")
    torch.save(model.state_dict(), save_model_path)
    print(f"[Model Saved] Fine-tuned model checkpoint saved to: {save_model_path}")

    # 6. Plot Loss & Accuracy Curves
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    epochs_range = range(1, EPOCHS + 1)
    plt.plot(epochs_range, history["train_loss"], 'o-', color='#0284c7', label="Training Loss", linewidth=2)
    plt.plot(epochs_range, history["val_loss"], 's--', color='#ef4444', label="Validation Loss", linewidth=2)
    plt.title("SegFormer-B0 Cross-Entropy Loss Curve", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # mIoU & Pixel Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["val_miou"], '^-', color='#10b981', label="Validation mIoU (%)", linewidth=2)
    plt.plot(epochs_range, history["val_pixel_acc"], 'd--', color='#f59e0b', label="Pixel Accuracy (%)", linewidth=2)
    plt.title("Validation mIoU & Accuracy Progress", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Percentage (%)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, "fine_tuning_curves.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Artifact] Training curve plot saved to: {plot_path}")

    # 7. Save Metrics JSON Report
    metrics_report = {
        "dataset": "RUGD Off-Road Dataset",
        "model": "SegFormer-B0 (Vision Transformer)",
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "training_time_seconds": total_time,
        "epochs": EPOCHS,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_miou": history["val_miou"][-1],
        "final_pixel_accuracy": history["val_pixel_acc"][-1],
        "history": history,
        "per_class_labels": CLASS_NAMES[1:]
    }

    report_path = os.path.join(outputs_dir, "fine_tuning_metrics.json")
    with open(report_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"[Artifact] Metrics report saved to: {report_path}")

    print("=" * 60)
    print("  ALL FINE-TUNING STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return metrics_report

if __name__ == "__main__":
    train_model()
