import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    resnet152,
    ResNet152_Weights,
    resnext50_32x4d,
    ResNeXt50_32X4D_Weights,
    resnext101_32x8d,
    ResNeXt101_32X8D_Weights,
    wide_resnet50_2,
    Wide_ResNet50_2_Weights,
    wide_resnet101_2,
    Wide_ResNet101_2_Weights,
)


# --------------------
# Global TestDataset definition
# --------------------
# (Used during training; for TTA we will re-load images from disk)
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = sorted(
            [
                f
                for f in os.listdir(img_dir)
                if f.lower().endswith(("jpg", "jpeg", "png", "bmp", "gif"))
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


# --------------------
# Model factory: create a ResNet/ResNeXt model with optional dropout
# --------------------
def create_resnet_model(model_name, num_classes, dropout_p=0.5):
    if model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
    elif model_name == "resnet101":
        weights = ResNet101_Weights.IMAGENET1K_V2
        model = resnet101(weights=weights)
    elif model_name == "resnet152":
        weights = ResNet152_Weights.IMAGENET1K_V2
        model = resnet152(weights=weights)
    elif model_name == "resnext50_32x4d":
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        model = resnext50_32x4d(weights=weights)
    elif model_name == "resnext101_32x8d":
        weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        model = resnext101_32x8d(weights=weights)
    elif model_name == "wide_resnet50_2":
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        model = wide_resnet50_2(weights=weights)
    elif model_name == "wide_resnet101_2":
        weights = Wide_ResNet101_2_Weights.IMAGENET1K_V2
        model = wide_resnet101_2(weights=weights)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if dropout_p != 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# --------------------
# Main function for ensemble inference with bagging and TTA
# --------------------
def main():
    test_dir = "./data/test"
    results_folder = "./models"
    num_classes = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Bag model info: (model_name, checkpoint_path, dropout_value)
    bag_models_info = [
        (
            "resnext101_32x8d",
            os.path.join(results_folder, "next_bag_5_best_epoch_19.pth"),
            0.5,
        ),
        (
            "resnext101_32x8d",
            os.path.join(results_folder, "model_epoch_22_91.pth"),
            0.5,
        ),
        (
            "resnext101_32x8d",
            os.path.join(results_folder, "next_bag_3_model_epoch_23.pth"),
            0.5,
        ),
        (
            "resnext101_32x8d",
            os.path.join(results_folder, "next_bag_4_model_epoch_24.pth"),
            0.5,
        ),
        (
            "resnext101_32x8d",
            os.path.join(results_folder, "next_bag_1_best_epoch_22.pth"),
            0.5,
        ),
    ]

    # Define test-time augmentation (TTA) transforms.
    # Two variants: the standard transform and one with horizontal flip.
    tta_transforms = [
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=1.0),  # Always flip
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    ]

    # Manually iterate over test images.
    test_image_files = sorted(
        [
            f
            for f in os.listdir(test_dir)
            if f.lower().endswith(("jpg", "jpeg", "png", "bmp", "gif"))
        ]
    )

    # Dictionary to accumulate softmax probabilities per test image.
    ensemble_probs = (
        {}
    )  # Key: filename without extension, Value: cumulative probability vector

    # For each bag model, load the checkpoint and perform TTA inference.
    for b, (model_name, model_path, dropout_value) in enumerate(
        bag_models_info
    ):
        print(
            f"\n[Ensemble] Running Bag {b+1} using {model_name}..."
        )
        model = create_resnet_model(
            model_name, num_classes, dropout_p=dropout_value
        )
        state_dict = torch.load(model_path, map_location=device)

        if dropout_value != 0:
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("fc.0."):
                    new_key = key.replace("fc.0.", "fc.1.")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        # If necessary, adjust state_dict keys (this is model-specific).
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # For each test image, perform TTA.
        for img_name in tqdm(
            test_image_files, desc=f"Bag {b+1} TTA Inference", leave=False
        ):
            img_path = os.path.join(test_dir, img_name)
            # Open the raw image (we don't use the TestDataset transform here)
            image = Image.open(img_path).convert("RGB")
            tta_probs = []
            for tta_transform in tta_transforms:
                input_tensor = tta_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.softmax(output, dim=1).cpu().numpy()[0]
                    tta_probs.append(prob)
            avg_prob = np.mean(tta_probs, axis=0)
            fn_no_ext = os.path.splitext(img_name)[0]
            # Accumulate the probability from this bag.
            if fn_no_ext not in ensemble_probs:
                ensemble_probs[fn_no_ext] = avg_prob
            else:
                ensemble_probs[fn_no_ext] += avg_prob

    # Average the probabilities over all bags.
    final_predictions = []
    # Assume that training folder names (sorted) represent class names.
    class_names = sorted(os.listdir("./data/train"))
    for fn, prob_sum in ensemble_probs.items():
        avg_prob_over_bags = prob_sum / len(bag_models_info)
        pred_idx = int(np.argmax(avg_prob_over_bags))
        pred_class = class_names[pred_idx]
        final_predictions.append((fn, pred_class))

    # Save final ensemble predictions to CSV.
    csv_path = os.path.join(results_folder, "ensemble_submission.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        for fn, pred_class in sorted(final_predictions):
            writer.writerow([fn, pred_class])
    print(f"\nEnsemble test predictions saved to: {csv_path}")


if __name__ == "__main__":
    main()
