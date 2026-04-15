import argparse
import os
from typing import Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from use_croma import PretrainedCROMA

from tqdm.auto import tqdm


# ---------------------------------------------------------------------
# Dataset So2Sat LCZ42 -> CROMA-compatible tensors
# ---------------------------------------------------------------------


def croma_normalize_sample(x: torch.Tensor,
                           use_8_bit: bool = True) -> torch.Tensor:
    """Normalize one sample (C, H, W) using the CROMA-style per-channel rule.

    This follows the reference implementation: per-channel min/max are
    computed as mean +/- 2 * std, values are clipped, then optionally
    scaled to [0, 255] uint8.
    """
    x = x.unsqueeze(0)  # (1, C, H, W)
    x = x.float()
    imgs = []
    for channel in range(x.shape[1]):
        channel_data = x[:, channel, :, :]  # (1, H, W)
        min_value = channel_data.mean() - 2 * channel_data.std()
        max_value = channel_data.mean() + 2 * channel_data.std()

        denom = max_value - min_value
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)

        if use_8_bit:
            img = (channel_data - min_value) / denom * 255.0
            img = torch.clip(img, 0, 255).unsqueeze(dim=1).to(torch.uint8)
        else:
            img = (channel_data - min_value) / denom
            img = torch.clip(img, 0, 1).unsqueeze(dim=1)

        imgs.append(img)

    x_norm = torch.cat(imgs, dim=1)  # (1, C, H, W)
    x_norm = x_norm.squeeze(0)       # (C, H, W)
    return x_norm


class So2SatLCZ42Dataset(Dataset):
    """Dataset for So2Sat LCZ42 with optional filtering of urban classes only.

    Steps:
      - Load sen1 (8 channels) and sen2 (10 channels) from an HDF5 file.
      - Build 2-channel SAR from sen1 bands 5 and 6 (Lee-filtered VH/VV).
      - Build 12-channel optical with the correct Sentinel-2 band ordering:
        B1 (zero), B2, B3, B4, B5, B6, B7, B8, B8A, B9 (zero), B11, B12.
      - Optionally keep only samples whose class index is in [0, 9]
        (urban classes LCZ 1–10).
      - Resize both SAR and optical to `image_resolution`.
      - Apply CROMA per-channel normalization (optionally to 8-bit).
      - Convert one-hot labels to integer class indices.
    """

    def __init__(
        self,
        h5_path: str,
        image_resolution: int = 120,
        normalize: bool = True,
        use_8_bit: bool = True,
        urban_only: bool = False,
    ) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.image_resolution = image_resolution
        self.normalize = normalize
        self.use_8_bit = use_8_bit
        self.urban_only = urban_only

        self._h5_file = None
        self._sen1 = None
        self._sen2 = None
        self._labels = None

        # Precompute the list of indices to keep and dataset length
        with h5py.File(self.h5_path, "r") as f:
            labels = f["label"][:]          # (N, 17)
            class_idx = np.argmax(labels, axis=1)  # (N,)

            if self.urban_only:
                # Keep only the 10 urban classes: LCZ 1–10 -> indices 0–9
                mask = class_idx < 10
                self.indices = np.where(mask)[0]
            else:
                self.indices = np.arange(labels.shape[0])

            self._length = len(self.indices)

    def _lazy_open(self) -> None:
        """Open the HDF5 file on first access, per worker."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._sen1 = self._h5_file["sen1"]
            self._sen2 = self._h5_file["sen2"]
            self._labels = self._h5_file["label"]

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def _build_sar(sen1: torch.Tensor) -> torch.Tensor:
        """Build the 2-channel SAR image from sen1."""
        sar = sen1[4:6, :, :]  # (2, H, W)
        return sar

    @staticmethod
    def _build_optical(sen2: torch.Tensor) -> torch.Tensor:
        """Build the 12-channel optical image in Sentinel-2 order."""
        _, height, width = sen2.shape
        zeros_band = torch.zeros(1, height, width, dtype=sen2.dtype)

        b2 = sen2[0:1, :, :]
        b3 = sen2[1:2, :, :]
        b4 = sen2[2:3, :, :]
        b5 = sen2[3:4, :, :]
        b6 = sen2[4:5, :, :]
        b7 = sen2[5:6, :, :]
        b8a = sen2[6:7, :, :]
        b11 = sen2[7:8, :, :]
        b12 = sen2[8:9, :, :]
        b8 = sen2[9:10, :, :]

        optical = torch.cat(
            [
                zeros_band,  # B1
                b2,          # B2
                b3,          # B3
                b4,          # B4
                b5,          # B5
                b6,          # B6
                b7,          # B7
                b8,          # B8
                b8a,         # B8A
                zeros_band,  # B9
                b11,         # B11
                b12,         # B12
            ],
            dim=0,
        )
        return optical

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        self._lazy_open()

        real_index = int(self.indices[index])

        sen1_np = self._sen1[real_index]        # (32, 32, 8)
        sen2_np = self._sen2[real_index]        # (32, 32, 10)
        label_one_hot = self._labels[real_index]

        sen1 = torch.from_numpy(sen1_np).permute(2, 0, 1).float()
        sen2 = torch.from_numpy(sen2_np).permute(2, 0, 1).float()

        sar = self._build_sar(sen1)          # (2, 32, 32)
        optical = self._build_optical(sen2)  # (12, 32, 32)

        if self.image_resolution is not None:
            sar = F.interpolate(
                sar.unsqueeze(0),
                size=(self.image_resolution, self.image_resolution),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            optical = F.interpolate(
                optical.unsqueeze(0),
                size=(self.image_resolution, self.image_resolution),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.normalize:
            sar = croma_normalize_sample(sar, use_8_bit=self.use_8_bit)
            optical = croma_normalize_sample(
                optical,
                use_8_bit=self.use_8_bit,
            )
            if self.use_8_bit:
                sar = sar.float() / 255.0
                optical = optical.float() / 255.0

        label_index = int(np.argmax(label_one_hot).astype(np.int64))
        return sar, optical, label_index

    def close(self) -> None:
        """Close the underlying HDF5 file if it is open."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._sen1 = None
            self._sen2 = None
            self._labels = None


# ---------------------------------------------------------------------
# CROMA backbone + LCZ classifier
# ---------------------------------------------------------------------


class CromaLCZClassifier(nn.Module):
    """CROMA backbone with a LCZ classification head on joint_GAP features."""

    def __init__(
        self,
        pretrained_path: str,
        num_classes: int = 17,
        size: str = "base",
        image_resolution: int = 120,
        device: str = "cuda",
        modality: str = "both",
    ) -> None:
        super().__init__()

        self.backbone = PretrainedCROMA(
            pretrained_path=pretrained_path,
            size=size,
            modality=modality,
            image_resolution=image_resolution,
        ).to(device)

        self.image_resolution = image_resolution
        self.device = device
        self.num_classes = num_classes

        with torch.no_grad():
            dummy_sar = torch.zeros(
                1, 2, image_resolution, image_resolution, device=device
            )
            dummy_optical = torch.zeros(
                1, 12, image_resolution, image_resolution, device=device
            )
            outputs = self.backbone(
                SAR_images=dummy_sar,
                optical_images=dummy_optical,
            )
            joint_gap = outputs["joint_GAP"]
            feature_dim = joint_gap.shape[1]

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self,
                sar: torch.Tensor,
                optical: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(SAR_images=sar, optical_images=optical)
        features = outputs["joint_GAP"]
        logits = self.classifier(features)
        return logits


def set_backbone_requires_grad(model: CromaLCZClassifier,
                               requires_grad: bool) -> None:
    """Enable or disable gradients for the CROMA backbone."""
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad


# ---------------------------------------------------------------------
# Class weights from HDF5
# ---------------------------------------------------------------------


def compute_class_weights_from_h5(
    h5_path: str,
    num_classes: int = 10,
    urban_only: bool = True,
) -> torch.Tensor:
    """Compute class weights from an HDF5 label dataset."""
    with h5py.File(h5_path, "r") as f:
        labels = f["label"][:]  # (N, 17)
        class_idx = np.argmax(labels, axis=1)  # (N,)

        if urban_only:
            mask = class_idx < num_classes
            class_idx = class_idx[mask]

        counts = np.bincount(class_idx, minlength=num_classes).astype(np.float64)

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    criterion: nn.Module,
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for sar, optical, labels in tqdm(
        dataloader,
        desc="Train",
        dynamic_ncols=True,
    ):
        sar = sar.to(device, non_blocking=True)
        optical = optical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(sar, optical)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    criterion: nn.Module,
    num_classes: int = 10,
) -> Tuple[float, float, float]:
    """Evaluate on a validation or test set.

    Returns:
        avg_loss, accuracy, f1_macro
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for sar, optical, labels in dataloader:
        sar = sar.to(device, non_blocking=True)
        optical = optical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(sar, optical)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()

        total_loss += loss.item() * labels.size(0)
        total_correct += correct

        indices = labels * num_classes + preds
        binc = torch.bincount(
            indices,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        confusion += binc.cpu()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)

    tp = torch.diag(confusion).float()
    per_class_pred = confusion.sum(dim=0).float()
    per_class_true = confusion.sum(dim=1).float()

    precision = tp / (per_class_pred + 1e-12)
    recall = tp / (per_class_true + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    f1_macro = f1.mean().item()

    return avg_loss, accuracy, f1_macro


# ---------------------------------------------------------------------
# Main: linear probing + finetuning
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune CROMA on So2Sat LCZ42."
    )
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, required=True)
    parser.add_argument("--test_h5", type=str, required=True)
    parser.add_argument("--croma_weights", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--img_res", type=int, default=120)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs_phase1", type=int, default=5)
    parser.add_argument("--epochs_phase2", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    num_classes = 10

    # Class weights from training HDF5 (urban-only)
    class_weights = compute_class_weights_from_h5(
        args.train_h5,
        num_classes=num_classes,
        urban_only=True,
    ).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    train_dataset = So2SatLCZ42Dataset(
        args.train_h5,
        image_resolution=args.img_res,
        urban_only=True,
    )
    val_dataset = So2SatLCZ42Dataset(
        args.val_h5,
        image_resolution=args.img_res,
        urban_only=True,
    )
    test_dataset = So2SatLCZ42Dataset(
        args.test_h5,
        image_resolution=args.img_res,
        urban_only=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = CromaLCZClassifier(
        pretrained_path=args.croma_weights,
        num_classes=num_classes,
        size="base",
        image_resolution=args.img_res,
        device=device,
        modality="both",
    ).to(device)

    # Losses
    criterion_train = nn.CrossEntropyLoss(weight=class_weights)
    criterion_eval = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1: linear probing (backbone frozen)
    set_backbone_requires_grad(model, False)
    optimizer_head = torch.optim.Adam(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    for epoch in range(args.epochs_phase1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer_head, device, criterion_train
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, device, criterion_eval, num_classes=num_classes
        )
        print(
            f"[Phase1] {epoch + 1}/{args.epochs_phase1} "
            f"loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1_macro={val_f1:.4f}"
        )

    # Phase 2: finetuning (backbone + head), with best model on f1_macro
    set_backbone_requires_grad(model, True)
    optimizer_ft = torch.optim.Adam(
        [
            {"params": model.backbone.parameters(), "lr": 5e-6},
            {"params": model.classifier.parameters(), "lr": 2e-4},
        ],
        weight_decay=2e-5,
    )

    best_f1 = -1.0
    best_state = None

    for epoch in range(args.epochs_phase2):
        train_loss = train_one_epoch(
            model, train_loader, optimizer_ft, device, criterion_train
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, device, criterion_eval, num_classes=num_classes
        )
        print(
            f"[Phase2] {epoch + 1}/{args.epochs_phase2} "
            f"loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1_macro={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()

    if best_state is not None:
        best_path = os.path.join(
            args.output_dir,
            "croma_lcz42_best_f1_macro.pth",
        )
        torch.save(best_state, best_path)
        print(f"Best model (F1_macro={best_f1:.4f}) saved to {best_path}")
        model.load_state_dict(best_state)

    test_loss, test_acc, test_f1 = evaluate(
        model, test_loader, device, criterion_eval, num_classes=num_classes
    )
    print(
        f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} "
        f"f1_macro={test_f1:.4f}"
    )

    save_full = os.path.join(
        args.output_dir,
        "croma_lcz42_finetuned_full.pth",
    )
    save_backbone = os.path.join(
        args.output_dir,
        "croma_backbone_finetuned.pth",
    )

    torch.save(model.state_dict(), save_full)
    torch.save(
        {
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
        },
        save_backbone,
    )

    train_dataset.close()
    val_dataset.close()
    test_dataset.close()


if __name__ == "__main__":
    main()
