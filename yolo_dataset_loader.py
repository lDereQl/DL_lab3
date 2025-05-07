import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# === Configuration ===
BASE_PATH = './filtered_coco_yolo'
IMG_PATHS = {
    'train': os.path.join(BASE_PATH, 'images/train'),
    'val': os.path.join(BASE_PATH, 'images/val')
}
LABEL_PATHS = {
    'train': os.path.join(BASE_PATH, 'labels/train'),
    'val': os.path.join(BASE_PATH, 'labels/val')
}
CLASS_MAPPING = {0: 'boat', 1: 'bird'}
IMG_SIZE = 640


# === YOLO Dataset Class ===
class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=IMG_SIZE, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, img_file.replace('.jpg', '.txt'))

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Load labels
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x1 = (x_center - width / 2) * w
                    y1 = (y_center - height / 2) * h
                    x2 = (x_center + width / 2) * w
                    y2 = (y_center + height / 2) * h
                    boxes.append([x1, y1, x2, y2, int(class_id)])

        # Convert to Tensor
        boxes = torch.tensor(boxes)

        # Resize the image
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.tensor(img).permute(2, 0, 1) / 255.0

        return img, boxes


# === Custom Collate Function ===
def custom_collate_fn(batch):
    images = []
    targets = []

    for img, boxes in batch:
        images.append(img)
        targets.append(boxes)

    # Stack images (always same size)
    images = torch.stack(images)

    # Return images and list of tensors for bounding boxes
    return images, targets


# === DataLoader Setup with Custom Collate ===
def get_dataloader(batch_size=8):
    train_dataset = YoloDataset(IMG_PATHS['train'], LABEL_PATHS['train'])
    val_dataset = YoloDataset(IMG_PATHS['val'], LABEL_PATHS['val'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader

# === Visualization Function ===
def visualize_sample(dataloader):
    images, targets = next(iter(dataloader))
    for i in range(min(2, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        for box in targets[i]:
            x1, y1, x2, y2, class_id = box
            # Convert tensor to integer before using as dictionary key
            class_id_int = int(class_id.item())
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none'))
            plt.text(x1, y1, CLASS_MAPPING[class_id_int], color='red')
        plt.show()


# === Main Execution ===
if __name__ == "__main__":
    train_loader, val_loader = get_dataloader()
    print("Loaded samples from train and val datasets.")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    visualize_sample(train_loader)