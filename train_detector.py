import torch
import os
import time
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from task1_ResNet import YOLOResNetDetector, BoatBirdDataset
from yolo_loss import YoloLoss
import torch.nn.utils.prune as prune


def apply_pruning(model, amount=0.3):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, 'weight')
    print("✅ Pruning applied to Conv2d layers")


def train_and_save(pretrained=True, save_dir="models", epochs=10):
    label = 'pretrained' if pretrained else 'scratch'

    # === Model and setup ===
    model = YOLOResNetDetector(pretrained=pretrained).to(device)
    criterion = YoloLoss(S=7, B=2, C=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    dataset = BoatBirdDataset('./filtered_coco_yolo', split='train')
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

    # === Training Loop ===
    model.train()
    losses = []
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"[{label}] Epoch {epoch + 1}/{epochs}")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

    # === Post-training Pruning ===
    apply_pruning(model)

    # === Save model ===
    duration = time.time() - start_time
    save_path = os.path.join(save_dir, f"detector_{label}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Saved {label} model to: {save_path}")
    print(f"⏱️ Training time: {duration:.2f} seconds\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_and_save(pretrained=True)
    train_and_save(pretrained=False)
