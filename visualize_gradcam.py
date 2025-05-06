import torch
import numpy as np
import cv2
import os
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from task1_ResNet import YOLOResNetDetector, BoatBirdDataset, val_collate_fn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.gradients = None
        self.activations = None

        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def save_activations(module, input, output):
            self.activations = output.detach()

        target_layer.register_forward_hook(save_activations)
        target_layer.register_full_backward_hook(save_gradients)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output = output[0]

        # We're focusing on one grid cell and class index (simplified)
        score = output[..., 5 + class_idx].max()
        score.backward()

        pooled_gradients = self.gradients.mean(dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max() if heatmap.max() != 0 else 1
        return heatmap


def overlay_heatmap(image: Image.Image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, image.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image.convert("RGB"))[:, :, ::-1]  # PIL to BGR
    overlay = cv2.addWeighted(heatmap_color, alpha, image_np, 1 - alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def visualize(model_path, class_idx, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    model = YOLOResNetDetector(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    target_layer = model.backbone[-1]  # Last conv layer in ResNet18
    gradcam = GradCAM(model, target_layer)

    dataset = BoatBirdDataset('./filtered_coco_yolo', split='val', return_original=True)
    indices = random.sample(range(len(dataset)), 5)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, collate_fn=val_collate_fn)

    for i, (img_tensor, _, originals, names) in enumerate(loader):
        img_tensor = img_tensor.to(device)
        heatmap = gradcam.generate(img_tensor, class_idx=class_idx)
        original_img = originals[0].resize((224, 224))
        result = overlay_heatmap(original_img, heatmap)

        Image.fromarray(result).save(f"{out_dir}/{label}_gradcam_{i+1}.png")
        print(f"✅ Saved: {label}_gradcam_{i+1}.png")


if __name__ == "__main__":
    visualize("models/detector_pretrained.pth", class_idx=0, out_dir="results1_gradcam", label="bird")
    visualize("models/detector_scratch.pth", class_idx=0, out_dir="results1_gradcam", label="bird_scratch")
    visualize("models/detector_pretrained.pth", class_idx=1, out_dir="results1_gradcam", label="boat")
    visualize("models/detector_scratch.pth", class_idx=1, out_dir="results1_gradcam", label="boat_scratch")
