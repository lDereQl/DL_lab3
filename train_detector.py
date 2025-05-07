import torch
import os
import time
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from task1_ResNet import YOLOResNetDetector, BoatBirdDataset
from yolo_loss import YoloLoss
import torch.nn.utils.prune as prune
from torch.cuda.amp import GradScaler, autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_pruning(model, amount=0.3):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, 'weight')
    print("✅ Pruning applied to Conv2d layers")


def get_max_batch_size(model, image_size=(416, 416)):
    """
    Automatically find the max batch size for the available GPU memory
    """
    batch_size = 8
    while True:
        try:
            dummy_input = torch.randn(batch_size, 3, *image_size).to(device)
            _ = model(dummy_input)
            batch_size += 8
        except RuntimeError:
            print(f"✅ Optimal Batch Size: {batch_size - 8}")
            return batch_size - 8


def train_and_save(pretrained=True, save_dir="models", epochs=20):
    label = 'pretrained' if pretrained else 'scratch'

    # === Model and setup ===
    model = YOLOResNetDetector(pretrained=pretrained).to(device)
    criterion = YoloLoss(S=7, B=2, C=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler()
    gradient_clip_value = 1.0

    # === Optimal Batch Size Calculation ===
    optimal_batch_size = get_max_batch_size(model)
    dataset = BoatBirdDataset('./filtered_coco_yolo', split='train')
    loader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # === Training Loop ===
    model.train()
    losses = []
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"[{label}] Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            with autocast():
                preds = model(imgs)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())

        scheduler.step(epoch_loss)

        # Apply pruning every 5 epochs
        if (epoch + 1) % 5 == 0:
            apply_pruning(model)

    # === Save model ===
    duration = time.time() - start_time
    save_path = os.path.join(save_dir, f"detector_{label}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Saved {label} model to: {save_path}")
    print(f"⏱️ Training time: {duration:.2f} seconds\n")


if __name__ == "__main__":
    train_and_save(pretrained=True)
    train_and_save(pretrained=False)
