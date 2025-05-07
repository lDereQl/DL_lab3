import torch
import torch.optim as optim
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from yolo_dataset_loader import get_dataloader
import matplotlib.pyplot as plt

# === Configuration ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2  # boat and bird
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
MODEL_PATH = 'yolov5_boat_bird.pth'

# === Load Datasets ===
train_loader, val_loader = get_dataloader(BATCH_SIZE)

# === Model Definition ===
model = fasterrcnn_resnet50_fpn(num_classes=NUM_CLASSES + 1)  # +1 for background class
model.to(DEVICE)

# === Optimizer and Scheduler ===
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# === Training Loop ===
def train_one_epoch():
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{
            "boxes": t[:, :4].to(DEVICE),
            "labels": (t[:, 4] + 1).to(DEVICE)  # +1 to avoid background class = 0
        } for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()

    scheduler.step()
    return epoch_loss / len(train_loader)


# === Validation Loop ===
def evaluate_model():
    model.eval()
    total_boxes = 0
    correct_boxes = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(DEVICE) for image in images)
            outputs = model(images)

            for idx, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu()
                true_boxes = targets[idx][:, :4]

                total_boxes += len(true_boxes)
                correct_boxes += sum(1 for pred in pred_boxes if any(
                    torch.allclose(pred, true, atol=5) for true in true_boxes
                ))

    precision = correct_boxes / total_boxes if total_boxes > 0 else 0
    recall = correct_boxes / len(val_loader.dataset)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    return precision, recall


# === Training Execution ===
if __name__ == "__main__":
    print("Starting training...")
    history = {'loss': [], 'precision': [], 'recall': []}

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        loss = train_one_epoch()
        precision, recall = evaluate_model()

        history['loss'].append(loss)
        history['precision'].append(precision)
        history['recall'].append(recall)

        # Save checkpoint
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved at {MODEL_PATH}")

    # Plot metrics
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Loss')
    plt.plot(history['precision'], label='Precision')
    plt.plot(history['recall'], label='Recall')
    plt.title("Training Metrics Over Time")
    plt.legend()
    plt.show()
