# evaluate_detector.py

import torch
from task1_ResNet import YOLOResNetDetector, BoatBirdDataset, val_collate_fn
import os
from torch.utils.data import DataLoader
import random
from PIL import ImageDraw, ImageFont

def test_and_visualize(model, dataset, outdir="results1_eval", n_samples=5, device="cpu"):
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, collate_fn=val_collate_fn)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    with torch.no_grad():
        for i, (img_tensor, _, originals, names) in enumerate(loader):
            img_tensor = img_tensor.to(device)
            pred = model(img_tensor)[0].cpu()
            img = originals[0].resize((224, 224)).copy()
            draw = ImageDraw.Draw(img)
            detected = False
            for gy in range(7):
                for gx in range(7):
                    for b in range(2):
                        offset = b * (5 + 2)
                        cell = pred[gy, gx]
                        conf = cell[offset + 4].item()
                        if conf > 0.3:
                            x = cell[offset + 0].item() * 224
                            y = cell[offset + 1].item() * 224
                            w = cell[offset + 2].item() * 224
                            h = cell[offset + 3].item() * 224
                            xmin = max(0, int(x - w / 2))
                            ymin = max(0, int(y - h / 2))
                            xmax = min(223, int(x + w / 2))
                            ymax = min(223, int(y + h / 2))
                            class_probs = cell[offset + 5:offset + 7].softmax(dim=0)
                            class_idx = torch.argmax(class_probs).item()
                            label = "bird" if class_idx == 0 else "boat"
                            color = "orange" if class_idx == 0 else "blue"
                            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
                            draw.text((xmin + 2, ymin - 10), f"{label}", fill="white", font=font)
                            print(f"Image: {names[0]} — Predicted: {label} (Confidence: {conf:.2f})")
                            detected = True
            if not detected:
                print(f"Image: {names[0]} — No confident prediction.")
            img.save(f"{outdir}/test{i+1}.png")


def evaluate_model(weights_path, label, outdir="results1_eval", device="cpu"):
    print(f"\n🔍 Evaluating model: {label}")

    model = YOLOResNetDetector(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    val_dataset = BoatBirdDataset('./filtered_coco_yolo', split='val', return_original=True)
    test_and_visualize(model, val_dataset, outdir=os.path.join(outdir, label), n_samples=5, device=device)


if __name__ == "__main__":
    evaluate_model("models/detector_pretrained.pth", label="pretrained")
    evaluate_model("models/detector_scratch.pth", label="scratch")
