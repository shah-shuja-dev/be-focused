"""
train.py -- FocusGuard CNN with full MLflow tracking

Usage:
    python scripts\train.py --lr 1e-3 --epochs 30 --batch_size 32
    python scripts\train.py --lr 5e-4 --epochs 50 --batch_size 64 --augment
"""

import argparse, os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch

# --- 1. Model ----------------------------------------------------------------


class FocusCNN(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# --- 2. Data loading ---------------------------------------------------------


def get_transforms(augment: bool):
    base = [
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    aug = [
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
    ]
    return transforms.Compose((aug if augment else []) + base), transforms.Compose(base)


def get_loaders(data_dir: str, batch_size: int, augment: bool, val_split: float = 0.2):
    train_t, val_t = get_transforms(augment)
    full = datasets.ImageFolder(data_dir, transform=train_t)
    classes = full.classes  # ['focused', 'not_focused']

    n_val = int(len(full) * val_split)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=val_t)

    # IMPORTANT: num_workers=0 on Windows to avoid DataLoader multiprocessing crashes
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, classes


# --- 3. Evaluation -----------------------------------------------------------


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, all_preds, all_labels = 0.0, 0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            total_loss += loss.item() * images.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    n = len(loader.dataset)
    return (
        total_loss / n,
        correct / n,
        f1_score(all_labels, all_preds, average="macro"),
        all_preds,
        all_labels,
    )


def save_confusion_matrix(preds, labels, classes, path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- 4. Main training loop ---------------------------------------------------


def train(args):
    device = torch.device("cpu")  # AMD R7 M340 has no ROCm support
    print(f"Device: {device}")

    train_loader, val_loader, classes = get_loaders(
        args.data_dir, args.batch_size, args.augment
    )
    print(f"Classes : {classes}")
    print(f"Train   : {len(train_loader.dataset)}")
    print(f"Val     : {len(val_loader.dataset)}")

    model = FocusCNN(dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Point MLflow at the tracking server we started separately
    mlflow.set_tracking_uri("http://localhost:5000")

    # Create or reuse the experiment named "focus_classifier"
    mlflow.set_experiment(args.experiment_name)

    # Everything inside this block = one Run in MLflow
    with mlflow.start_run(run_name=args.run_name) as run:
        print(f"\nMLflow run ID : {run.info.run_id}")
        print(f"Open UI at    : http://localhost:5000")

        # log_params() stores inputs to training
        # These become filterable columns in the UI
        mlflow.log_params(
            {
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "augment": args.augment,
                "optimizer": "adam",
                "scheduler": "cosine_annealing",
                "architecture": "FocusCNN_v1",
                "image_size": 96,
                "n_train": len(train_loader.dataset),
                "n_val": len(val_loader.dataset),
            }
        )

        # Tags = free-form labels for grouping/filtering runs
        mlflow.set_tags({"dataset_version": "v1", "stage": "experiment"})

        best_val_acc = 0.0
        best_model_path = os.path.join("models", "best_model.pt")
        os.makedirs("models", exist_ok=True)
        history = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss, train_correct = 0.0, 0
            t0 = time.time()

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                train_correct += (out.argmax(1) == labels).sum().item()

            scheduler.step()
            train_loss /= len(train_loader.dataset)
            train_acc = train_correct / len(train_loader.dataset)

            val_loss, val_acc, val_f1, preds, true_labels = evaluate(
                model, val_loader, criterion, device
            )
            elapsed = time.time() - t0

            # log_metrics() with step=epoch -> plotted as time series in UI
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch_time": elapsed,
                },
                step=epoch,
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 4),
                    "val_f1": round(val_f1, 4),
                }
            )

            print(
                f"Epoch {epoch:03d}/{args.epochs}  "
                f"train={train_loss:.4f}/{train_acc:.4f}  "
                f"val={val_loss:.4f}/{val_acc:.4f}  f1={val_f1:.4f}  {elapsed:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"  ^ New best val_acc={val_acc:.4f}")

        # Log artifacts -- files stored alongside the run
        cm_path = os.path.join("models", "confusion_matrix.png")
        save_confusion_matrix(preds, true_labels, classes, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        history_path = os.path.join("models", "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        mlflow.log_artifact(history_path, artifact_path="data")

        report_path = os.path.join("models", "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(classification_report(true_labels, preds, target_names=classes))
        mlflow.log_artifact(report_path, artifact_path="reports")

        # Load best model weights
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # Get sample input for model signature
        sample_batch, _ = next(iter(val_loader))
        example_input = sample_batch[:1].to(device)

        # Log model with pickle format (avoids torch.export issues)
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=args.register_as,
            input_example=example_input.cpu().numpy(),
            serialization_format="pickle",  # Use pickle to avoid pt2 export issues
        )

        mlflow.log_metrics({"best_val_acc": best_val_acc, "final_val_f1": val_f1})

        print(f"\nDone. best_val_acc={best_val_acc:.4f}  run_id={run.info.run_id}")
    return run.info.run_id


# --- 5. Args -----------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"data\processed")
    ap.add_argument("--experiment_name", default="focus_classifier")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--register_as", default="FocusClassifier")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()
    train(args)
