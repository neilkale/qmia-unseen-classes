import os
import argparse
import torch
from torch.utils.data import DataLoader

from torch_models import get_model
from data_utils import CustomDataModule


def infer_num_classes(dataset_name: str) -> int:
    ds = dataset_name.lower()
    if "cifar100" in ds:
        return 100
    if "imagenet-1k" in ds or "imagenet_1k" in ds or "imagenet" in ds:
        return 1000
    if "cifar20" in ds:
        return 20
    return 10


def load_base_model(architecture: str, num_classes: int, base_model_dir: str, device: torch.device) -> torch.nn.Module:
    model = get_model(architecture, num_classes, freeze_embedding=False).to(device)
    state_dict = torch.load(os.path.join(base_model_dir, "model.pickle"), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for samples, targets, base_samples in loader:
            base_samples = base_samples.to(device)
            targets = targets.to(device)
            logits = model(base_samples)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--base_architecture", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default="./data/")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--image_size", type=int, default=-1)
    ap.add_argument("--base_image_size", type=int, default=-1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = infer_num_classes(args.dataset)
    model = load_base_model(args.base_architecture, num_classes, args.base_model_dir, device)

    dm = CustomDataModule(
        dataset_name=args.dataset,
        stage="eval",
        batch_size=args.batch_size,
        num_workers=8,
        image_size=args.image_size,
        base_image_size=args.base_image_size,
        data_root=args.data_root,
    )
    dm.setup()

    # In eval stage: test_dataset = private, val_dataset = test (public holdout)
    private_loader = dm.predict_dataloader()[0]
    public_loader = dm.predict_dataloader()[1]

    priv_acc = compute_accuracy(model, private_loader, device)
    pub_acc = compute_accuracy(model, public_loader, device)

    print(f"Base model dir: {args.base_model_dir}")
    print(f"Private (Train) accuracy: {priv_acc:.4f}")
    print(f"Public (Test) accuracy:  {pub_acc:.4f}")


if __name__ == "__main__":
    main() 