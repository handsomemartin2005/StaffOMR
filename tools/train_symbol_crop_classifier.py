from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_image_path(file_name: str, image_root: Path) -> Path:
    path = Path(file_name)
    if path.is_absolute() and path.exists():
        return path
    candidate = image_root / file_name
    if candidate.exists():
        return candidate
    candidate = image_root / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve image: {file_name}")


def clamp_crop(box: list[float], width: int, height: int, pad_ratio: float, pad_px: int) -> tuple[int, int, int, int] | None:
    x, y, w, h = [float(value) for value in box]
    pad = max(float(pad_px), max(w, h) * pad_ratio)
    x0 = max(0, int(x - pad))
    y0 = max(0, int(y - pad))
    x1 = min(width, int(x + w + pad + 0.999))
    y1 = min(height, int(y + h + pad + 0.999))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def load_coco_items(
    coco_path: Path,
    *,
    image_root: Path,
    max_per_class: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    coco = read_json(coco_path)
    images = {int(image["id"]): image for image in coco.get("images", [])}
    image_paths = {
        image_id: str(resolve_image_path(str(image["file_name"]), image_root))
        for image_id, image in images.items()
    }
    categories = {int(cat["id"]): str(cat["name"]) for cat in coco.get("categories", [])}
    by_class: dict[str, list[dict[str, Any]]] = {name: [] for _, name in sorted(categories.items())}
    for ann in coco.get("annotations", []):
        class_name = categories.get(int(ann["category_id"]))
        image_id = int(ann["image_id"])
        image = images.get(image_id)
        if class_name is None or image is None:
            continue
        by_class.setdefault(class_name, []).append(
            {
                "image_path": image_paths[image_id],
                "bbox": ann["bbox"],
                "width": int(image["width"]),
                "height": int(image["height"]),
                "class": class_name,
            }
        )
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    raw_counts = {name: len(items) for name, items in by_class.items()}
    for class_name, items in sorted(by_class.items()):
        rng.shuffle(items)
        selected.extend(items[: min(max_per_class, len(items))])
    rng.shuffle(selected)
    class_names = [name for _, name in sorted(categories.items()) if raw_counts.get(name, 0) > 0]
    return selected, class_names, raw_counts


class ImageCache:
    def __init__(self, limit: int = 24) -> None:
        self.limit = max(1, int(limit))
        self.cache: OrderedDict[str, Image.Image] = OrderedDict()

    def get(self, path: str) -> Image.Image:
        if path in self.cache:
            image = self.cache.pop(path)
            self.cache[path] = image
            return image
        image = Image.open(path).convert("RGB")
        self.cache[path] = image
        while len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return image


class CropDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, Any]],
        class_to_idx: dict[str, int],
        *,
        image_size: int,
        pad_ratio: float,
        pad_px: int,
        train: bool,
        scan_augment: bool = False,
        image_cache_limit: int = 24,
    ) -> None:
        self.items = items
        self.class_to_idx = class_to_idx
        self.pad_ratio = pad_ratio
        self.pad_px = pad_px
        self.cache = ImageCache(limit=image_cache_limit)
        transforms: list[Any] = [
            T.Resize((image_size, image_size)),
        ]
        if train:
            transforms.extend(
                [
                    T.RandomAffine(degrees=4, translate=(0.04, 0.04), scale=(0.88, 1.12), fill=255),
                ]
            )
            if scan_augment:
                transforms.extend(
                    [
                        T.RandomGrayscale(p=0.85),
                        T.RandomAutocontrast(p=0.35),
                        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                        T.ColorJitter(brightness=0.18, contrast=0.28),
                    ]
                )
            else:
                transforms.append(T.ColorJitter(brightness=0.08, contrast=0.08))
        transforms.append(T.ToTensor())
        if train and scan_augment:
            transforms.append(RandomScanNoise(p=0.45, sigma=0.035))
        transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transform = T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        image = self.cache.get(item["image_path"])
        crop = clamp_crop(item["bbox"], item["width"], item["height"], self.pad_ratio, self.pad_px)
        if crop is None:
            crop_image = Image.new("RGB", (16, 16), "white")
        else:
            crop_image = image.crop(crop)
        label = self.class_to_idx[item["class"]]
        return self.transform(crop_image), torch.tensor(label, dtype=torch.long)


class RandomScanNoise:
    def __init__(self, p: float = 0.4, sigma: float = 0.03) -> None:
        self.p = float(p)
        self.sigma = float(sigma)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.sigma
        return (tensor + noise).clamp(0.0, 1.0)


class SmallCropCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.15), nn.Linear(160, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def class_weights(items: list[dict[str, Any]], class_names: list[str], device: torch.device) -> torch.Tensor:
    counts = Counter(item["class"] for item in items)
    weights = []
    for name in class_names:
        weights.append(1.0 / max(1, counts.get(name, 0)))
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return tensor * (len(class_names) / tensor.sum().clamp_min(1e-6))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        total += int(labels.numel())
        correct += int((predictions == labels).sum().item())
        total_loss += float(loss.item()) * int(labels.numel())
        for truth, pred in zip(labels.cpu(), predictions.cpu()):
            confusion[int(truth), int(pred)] += 1
    per_class = []
    for index in range(num_classes):
        row = confusion[index]
        support = int(row.sum().item())
        correct_class = int(row[index].item())
        per_class.append(correct_class / support if support else 0.0)
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "per_class_accuracy": per_class,
        "confusion": confusion.tolist(),
    }


def write_confusion_csv(path: Path, class_names: list[str], confusion: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth\\pred", *class_names])
        for name, row in zip(class_names, confusion):
            writer.writerow([name, *row])


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Symbol Crop Classifier",
        "",
        f"Train items: {payload['train_items']}",
        f"Val items: {payload['val_items']}",
        f"Classes: {len(payload['class_names'])}",
        f"Best val accuracy: {payload['best_val_accuracy']:.4f}",
        "",
        "| Class | Val support | Accuracy |",
        "| --- | ---: | ---: |",
    ]
    final_eval = payload["final_eval"]
    confusion = final_eval["confusion"]
    for idx, name in enumerate(payload["class_names"]):
        support = sum(confusion[idx])
        lines.append(f"| {name} | {support} | {final_eval['per_class_accuracy'][idx]:.4f} |")
    lines.extend(
        [
            "",
            "## Focus Classes",
            "",
            "| Class | Accuracy | Common confusions |",
            "| --- | ---: | --- |",
        ]
    )
    focus = {"filled_notehead", "open_notehead", "sharp", "flat", "natural", "rest"}
    for idx, name in enumerate(payload["class_names"]):
        if name not in focus:
            continue
        row = confusion[idx]
        ranked = sorted(
            [(payload["class_names"][j], count) for j, count in enumerate(row) if j != idx and count],
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        confusions = ", ".join(f"{klass}:{count}" for klass, count in ranked) or "-"
        lines.append(f"| {name} | {final_eval['per_class_accuracy'][idx]:.4f} | {confusions} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small symbol crop classifier from COCO bbox crops.")
    parser.add_argument("--train-coco", type=Path, default=Path("outputs/v2_deim_ds_all/annotations/instances_train.json"))
    parser.add_argument("--val-coco", type=Path, default=Path("outputs/v2_deim_ds_all/annotations/instances_val.json"))
    parser.add_argument("--image-root", type=Path, default=Path("ds2_dense/ds2_dense/images"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/symbol_crop_classifier"))
    parser.add_argument("--max-train-per-class", type=int, default=500)
    parser.add_argument("--max-val-per-class", type=int, default=200)
    parser.add_argument("--classes", nargs="*", help="Optional class subset to train/evaluate.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pad-ratio", type=float, default=0.18)
    parser.add_argument("--pad-px", type=int, default=4)
    parser.add_argument("--scan-augment", action="store_true")
    parser.add_argument(
        "--image-cache-limit",
        type=int,
        default=24,
        help="Maximum number of decoded source pages cached while drawing symbol crops.",
    )
    parser.add_argument(
        "--group-by-image",
        action="store_true",
        help="Group sampled crops by source page to avoid repeatedly decoding large score images.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_items, train_class_names, raw_train_counts = load_coco_items(
        args.train_coco,
        image_root=args.image_root,
        max_per_class=args.max_train_per_class,
        seed=args.seed,
    )
    val_items, val_class_names, raw_val_counts = load_coco_items(
        args.val_coco,
        image_root=args.image_root,
        max_per_class=args.max_val_per_class,
        seed=args.seed + 1,
    )
    class_names = [name for name in train_class_names if name in set(val_class_names)]
    if args.classes:
        requested = set(args.classes)
        class_names = [name for name in class_names if name in requested]
        missing = sorted(requested - set(class_names))
        if missing:
            raise SystemExit(f"Requested classes missing from train/val COCO: {', '.join(missing)}")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_items = [item for item in train_items if item["class"] in class_to_idx]
    val_items = [item for item in val_items if item["class"] in class_to_idx]
    if args.group_by_image:
        train_items.sort(key=lambda item: (item["image_path"], item["class"], tuple(item["bbox"])))
        val_items.sort(key=lambda item: (item["image_path"], item["class"], tuple(item["bbox"])))

    device = torch.device(args.device)
    train_dataset = CropDataset(
        train_items,
        class_to_idx,
        image_size=args.image_size,
        pad_ratio=args.pad_ratio,
        pad_px=args.pad_px,
        train=True,
        scan_augment=args.scan_augment,
        image_cache_limit=args.image_cache_limit,
    )
    val_dataset = CropDataset(
        val_items,
        class_to_idx,
        image_size=args.image_size,
        pad_ratio=args.pad_ratio,
        pad_px=args.pad_px,
        train=False,
        image_cache_limit=args.image_cache_limit,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=not args.group_by_image, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SmallCropCNN(len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_items, class_names, device))

    history = []
    best_state = None
    best_val_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            predictions = logits.argmax(dim=1)
            total += int(labels.numel())
            correct += int((predictions == labels).sum().item())
            running_loss += float(loss.item()) * int(labels.numel())
        train_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, total),
            "train_accuracy": correct / max(1, total),
        }
        val_row = evaluate(model, val_loader, device, len(class_names))
        row = {**train_row, "val_loss": val_row["loss"], "val_accuracy": val_row["accuracy"]}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_row["accuracy"] > best_val_accuracy:
            best_val_accuracy = float(val_row["accuracy"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_eval = evaluate(model, val_loader, device, len(class_names))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "symbol_crop_classifier.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "class_names": class_names,
            "image_size": args.image_size,
            "pad_ratio": args.pad_ratio,
            "pad_px": args.pad_px,
            "architecture": "SmallCropCNN",
        },
        model_path,
    )
    confusion_csv = args.out_dir / "confusion_matrix.csv"
    write_confusion_csv(confusion_csv, class_names, final_eval["confusion"])
    payload = {
        "train_coco": str(args.train_coco),
        "val_coco": str(args.val_coco),
        "image_root": str(args.image_root),
        "model_path": str(model_path),
        "confusion_csv": str(confusion_csv),
        "train_items": len(train_items),
        "val_items": len(val_items),
        "class_names": class_names,
        "image_cache_limit": args.image_cache_limit,
        "group_by_image": args.group_by_image,
        "raw_train_counts": raw_train_counts,
        "raw_val_counts": raw_val_counts,
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "final_eval": final_eval,
    }
    summary_json = args.out_dir / "crop_classifier_summary.json"
    summary_md = args.out_dir / "crop_classifier_summary.md"
    write_json(summary_json, payload)
    summary_md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
                "model": str(model_path),
                "best_val_accuracy": best_val_accuracy,
                "final_val_accuracy": final_eval["accuracy"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
