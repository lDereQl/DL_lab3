import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from PIL import Image

from torchvision.models import resnet50, ResNet50_Weights
from yolo_loss import YoloLoss


class YOLOResNetDetector(nn.Module):
    def __init__(self, pretrained=True, grid_size=7, num_boxes=2, num_classes=2):
        super(YOLOResNetDetector, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.output_dim = num_boxes * (5 + num_classes)

        if pretrained:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet50(weights=None)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.head = nn.Conv2d(2048, self.output_dim, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.permute(0, 2, 3, 1)


class BoatBirdDataset(Dataset):
    def __init__(self, root, split='train', transform=None, return_original=False):
        self.img_dir = os.path.join(root, 'images', split)
        self.lbl_dir = os.path.join(root, 'labels', split)
        self.imgs = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.grid_size = 7
        self.num_boxes = 2
        self.num_classes = 2
        self.return_original = return_original

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        label_path = os.path.join(self.lbl_dir, self.imgs[idx].replace('.jpg', '.txt'))

        image = Image.open(img_path).convert('RGB')
        tensor_img = self.transform(image)

        target = torch.zeros((self.grid_size, self.grid_size, self.num_boxes * (5 + self.num_classes)))
        try:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    grid_x = int(x * self.grid_size)
                    grid_y = int(y * self.grid_size)
                    box = [x, y, w, h, 1.0] + [0] * self.num_classes
                    box[5 + int(cls)] = 1
                    target[grid_y, grid_x, :len(box)] = torch.tensor(box)
        except FileNotFoundError:
            pass

        if self.return_original:
            return tensor_img, target, image, self.imgs[idx]
        else:
            return tensor_img, target


def val_collate_fn(batch):
    images, targets, originals, names = zip(*batch)
    return torch.stack(images), torch.stack(targets), originals, names


# Optional: Create loss globally if training/evaluation needs it
yolo_loss = YoloLoss(S=7, B=2, C=2)
