import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import (
    resnext101_32x8d,
    ResNeXt101_32X8D_Weights,
)


# --------------------
# Global TestDataset definition
# --------------------
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # List test image files (all images in the folder)
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
# Helper functions for training and validation
# --------------------
def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# --------------------
# Main training function using bagging and early stopping
# --------------------
def main():
    # Directories and hyperparameters
    train_dir = "./data/train"
    val_dir = "./data/val"
    test_dir = "./data/test"
    num_classes = 100  # Adjust as needed
    num_epochs = 40
    batch_size = 32
    num_bags = 5  # Number of bagging models
    early_stop_patience = (
        6  # Stop if no improvement in val accuracy for these many epochs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    results_folder = "./results"
    os.makedirs(results_folder, exist_ok=True)

    # --------------------
    # Data Transforms
    # --------------------
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),  # more aggressive cropping
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),  # increased rotation angle
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # stronger color jitter
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    test_transform = val_transform

    # --------------------
    # Datasets and DataLoaders
    # --------------------
    num_workers = 8
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir, transform=train_transform
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_dir, transform=val_transform
    )
    test_dataset = TestDataset(test_dir, transform=test_transform)

    # For bagging, create a bootstrapped training dataset for each bag.
    bag_checkpoint_paths = []  # To store the best checkpoint for each bag
    bag_loss_histories = []  # Optionally track loss histories

    for bag in range(num_bags):
        print(f"\n=== Training Bag {bag+1}/{num_bags} ===")
        num_train = len(train_dataset)
        bag_indices = [
            random.choice(range(num_train)) for _ in range(num_train)
        ]
        bag_train_dataset = Subset(train_dataset, bag_indices)
        train_loader = DataLoader(
            bag_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        model = resnext101_32x8d(
            weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        )
        model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(model.fc.in_features, 100)
        )
        model = model.to(device)
        optimizer = optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )

        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        loss_history = []

        for epoch in range(num_epochs):
            print(f"Bag {bag+1} Epoch {epoch+1}")
            train_loss, train_acc = train_one_epoch(
                model, optimizer, train_loader, device
            )
            val_loss, val_acc = validate_one_epoch(model, val_loader, device)
            loss_history.append((train_loss, val_loss, train_acc, val_acc))
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            scheduler.step()

            # Early stopping based on best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                checkpoint_path = os.path.join(
                    results_folder, f"bag_{bag+1}_best_epoch_{best_epoch}.pth"
                )
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{early_stop_patience}")
                if patience_counter >= early_stop_patience:
                    print(
                        f"Early stopping triggered for Bag \
                        {bag+1} at epoch {epoch+1}"
                    )
                    break

        bag_checkpoint_paths.append(checkpoint_path)
        bag_loss_histories.append(loss_history)

    # --------------------
    # Ensemble Test Inference
    # --------------------
    print("\nPerforming ensemble test inference...")
    # Get class names from training dataset.
    class_names = train_dataset.classes
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    ensemble_probs = (
        {}
    )  # Dictionary to accumulate softmax probabilities per test image.

    for bag in range(num_bags):
        print(f"Loading Bag {bag+1} checkpoint for inference...")
        model = resnext101_32x8d(
            weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        )
        model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes)
        )
        model.load_state_dict(
            torch.load(bag_checkpoint_paths[bag], map_location=device)
        )
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for images, filenames in tqdm(
                test_loader, desc=f"Bag {bag+1} Inference", leave=False
            ):
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                for i, fn in enumerate(filenames):
                    fn_no_ext = os.path.splitext(fn)[0]
                    if fn_no_ext not in ensemble_probs:
                        ensemble_probs[fn_no_ext] = probs[i]
                    else:
                        ensemble_probs[fn_no_ext] += probs[i]

    # Average the probabilities over all bags and determine final predictions.
    final_predictions = {}
    for fn, prob_sum in ensemble_probs.items():
        avg_prob = prob_sum / num_bags
        pred_idx = int(np.argmax(avg_prob))
        final_predictions[fn] = class_names[pred_idx]

    # Save ensemble predictions to CSV.
    csv_path = os.path.join(results_folder, "ensemble_submission.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        for fn in sorted(final_predictions.keys()):
            writer.writerow([fn, final_predictions[fn]])
    print(f"Ensemble test predictions saved to: {csv_path}")

    # Optionally, plot training and validation loss curves for each bag.
    plt.figure()
    for bag in range(num_bags):
        history = bag_loss_histories[bag]
        epochs = range(1, len(history) + 1)
        train_losses_b = [h[0] for h in history]
        val_losses_b = [h[1] for h in history]
        plt.plot(epochs, train_losses_b, label=f"Bag {bag+1} Train")
        plt.plot(epochs, val_losses_b, label=f"Bag {bag+1} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Bagged Training and Validation Losses")
    plt.legend()
    plot_path = os.path.join(results_folder, "bagging_loss_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to: {plot_path}")


if __name__ == "__main__":
    # Define the global criterion (used in the helper functions)
    global criterion
    criterion = nn.CrossEntropyLoss()
    main()
