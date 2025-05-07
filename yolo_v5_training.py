import torch
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from yolo_dataset_loader import get_dataloader
import matplotlib.pyplot as plt

# === Configuration ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2  # boat and bird
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
BATCH_SIZE = 8

# === Load Datasets ===
train_loader, val_loader = get_dataloader(BATCH_SIZE)


# === Training Loop ===
def train_one_epoch(model, optimizer, scheduler):
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
def evaluate_model(model):
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
def run_training(mode):
    print(f"\n========== Training in {mode.upper()} Mode ==========")

    if mode == 'pretrained':
        print("Loading pre-trained weights from COCO...")
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        print("Training from scratch with random weights...")
        model = fasterrcnn_resnet50_fpn(pretrained=False)

    # Adjust for 2 classes (+1 for background)
    model.roi_heads.box_predictor = torch.nn.Sequential(
        torch.nn.Linear(1024, NUM_CLASSES + 1),
        torch.nn.Softmax(dim=1)
    )

    model.to(DEVICE)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # History Tracking
    history = {'loss': [], 'precision': [], 'recall': []}

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        loss = train_one_epoch(model, optimizer, scheduler)
        precision, recall = evaluate_model(model)

        history['loss'].append(loss)
        history['precision'].append(precision)
        history['recall'].append(recall)

    # Save Model
    model_path = f'yolov5_boat_bird_{mode}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    return history


# === Run Both Modes ===
history_pretrained = run_training('pretrained')
history_scratch = run_training('scratch')

# === Plot Comparison ===
plt.figure(figsize=(10, 5))
plt.plot(history_pretrained['loss'], label='Pretrained - Loss', linestyle='--')
plt.plot(history_scratch['loss'], label='Scratch - Loss')
plt.plot(history_pretrained['precision'], label='Pretrained - Precision', linestyle='--')
plt.plot(history_scratch['precision'], label='Scratch - Precision')
plt.plot(history_pretrained['recall'], label='Pretrained - Recall', linestyle='--')
plt.plot(history_scratch['recall'], label='Scratch - Recall')
plt.title("Training Metrics Comparison")
plt.legend()
plt.show()
