"""
Conditional GAN pipeline for minority-class image synthesis on APTOS 2019.

This module trains a class-conditional GAN and generates synthetic samples
for minority DR grades to reduce class imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


@dataclass
class CGANConfig:
    num_classes: int = 5
    image_size: int = 64
    channels: int = 3
    latent_dim: int = 128
    embedding_dim: int = 64
    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    label_smoothing_real: float = 0.9
    device: str = "cuda"
    checkpoint_interval: int = 10


class APTOSCGANDataset(Dataset):
    """Dataset wrapper for class-conditional GAN training."""

    SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        image_size: int = 64,
        file_extension: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        self.file_extension = file_extension or self._detect_extension()

    def _detect_extension(self) -> str:
        if len(self.df) == 0:
            return ".png"

        sample = str(self.df.iloc[0]["id_code"])
        for ext in self.SUPPORTED_EXTENSIONS:
            if (self.image_dir / f"{sample}{ext}").exists():
                return ext
        return ".png"

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, image_id: str) -> Path:
        path = self.image_dir / f"{image_id}{self.file_extension}"
        if path.exists():
            return path

        for ext in self.SUPPORTED_EXTENSIONS:
            candidate = self.image_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Image not found for id_code={image_id}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["id_code"])
        label = int(row["diagnosis"])

        image_path = self._resolve_path(image_id)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 127.5 - 1.0
        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class ConditionalGenerator(nn.Module):
    """DCGAN-style conditional generator."""

    def __init__(self, latent_dim: int, num_classes: int, embedding_dim: int, channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, embedding_dim)

        self.proj = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels)
        x = torch.cat([z, emb], dim=1)
        x = self.proj(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    """DCGAN-style conditional discriminator."""

    def __init__(self, num_classes: int, image_size: int = 64, channels: int = 3):
        super().__init__()
        self.image_size = image_size
        self.label_emb = nn.Embedding(num_classes, image_size * image_size)

        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.label_emb(labels).view(labels.size(0), 1, self.image_size, self.image_size)
        x = torch.cat([images, label_map], dim=1)
        out = self.net(x)
        return out.view(-1)


class ConditionalGANTrainer:
    """Trainer for class-conditional GAN used in minority sample generation."""

    def __init__(self, config: CGANConfig):
        self.config = config
        if self.config.image_size != 64:
            raise ValueError(
                "Current cGAN architecture supports image_size=64 only. "
                f"Received {self.config.image_size}."
            )

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.generator = ConditionalGenerator(
            latent_dim=config.latent_dim,
            num_classes=config.num_classes,
            embedding_dim=config.embedding_dim,
            channels=config.channels,
        ).to(self.device)

        self.discriminator = ConditionalDiscriminator(
            num_classes=config.num_classes,
            image_size=config.image_size,
            channels=config.channels,
        ).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )

    def train(self, loader: DataLoader, output_dir: Optional[Path] = None) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {"g_loss": [], "d_loss": []}
        output_dir = Path(output_dir) if output_dir is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            g_running = 0.0
            d_running = 0.0
            steps = 0

            for real_images, labels in loader:
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)
                batch_size = real_images.size(0)

                real_targets = torch.full(
                    (batch_size,),
                    fill_value=self.config.label_smoothing_real,
                    device=self.device,
                )
                fake_targets = torch.zeros(batch_size, device=self.device)

                # Train discriminator
                self.opt_d.zero_grad(set_to_none=True)

                logits_real = self.discriminator(real_images, labels)
                loss_real = self.criterion(logits_real, real_targets)

                z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                fake_images = self.generator(z, labels)
                logits_fake = self.discriminator(fake_images.detach(), labels)
                loss_fake = self.criterion(logits_fake, fake_targets)

                d_loss = 0.5 * (loss_real + loss_fake)
                d_loss.backward()
                self.opt_d.step()

                # Train generator
                self.opt_g.zero_grad(set_to_none=True)
                z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                gen_images = self.generator(z, labels)
                gen_logits = self.discriminator(gen_images, labels)
                g_loss = self.criterion(gen_logits, torch.ones(batch_size, device=self.device))
                g_loss.backward()
                self.opt_g.step()

                g_running += float(g_loss.item())
                d_running += float(d_loss.item())
                steps += 1

            g_epoch = g_running / max(steps, 1)
            d_epoch = d_running / max(steps, 1)
            history["g_loss"].append(g_epoch)
            history["d_loss"].append(d_epoch)

            print(
                f"[cGAN] Epoch {epoch:03d}/{self.config.epochs} | "
                f"D Loss: {d_epoch:.4f} | G Loss: {g_epoch:.4f}"
            )

            if output_dir is not None and (
                epoch % self.config.checkpoint_interval == 0 or epoch == self.config.epochs
            ):
                ckpt = {
                    "epoch": epoch,
                    "generator": self.generator.state_dict(),
                    "discriminator": self.discriminator.state_dict(),
                    "config": self.config.__dict__,
                    "history": history,
                }
                torch.save(ckpt, output_dir / f"cgan_epoch_{epoch:03d}.pth")

        return history

    @torch.no_grad()
    def generate_images(
        self,
        class_id: int,
        num_images: int,
        out_dir: Path,
        prefix: str,
        save_size: int = 224,
    ) -> List[str]:
        self.generator.eval()
        out_dir.mkdir(parents=True, exist_ok=True)

        if num_images <= 0:
            return []

        labels = torch.full((num_images,), int(class_id), dtype=torch.long, device=self.device)
        z = torch.randn(num_images, self.config.latent_dim, device=self.device)
        fake = self.generator(z, labels)
        fake = ((fake.clamp(-1, 1) + 1.0) * 127.5).byte().cpu().numpy()

        generated_ids: List[str] = []
        for idx in range(num_images):
            arr = np.transpose(fake[idx], (1, 2, 0))
            if save_size != self.config.image_size:
                arr = cv2.resize(arr, (save_size, save_size), interpolation=cv2.INTER_CUBIC)
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            image_id = f"{prefix}_class{class_id}_{idx:05d}"
            image_path = out_dir / f"{image_id}.png"
            cv2.imwrite(str(image_path), img_bgr)
            generated_ids.append(image_id)

        return generated_ids


def load_training_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "id_code" not in df.columns and "image" in df.columns:
        df = df.rename(columns={"image": "id_code"})
    if "diagnosis" not in df.columns:
        raise ValueError("Training CSV must contain diagnosis column")
    return df


def compute_generation_plan(
    df: pd.DataFrame,
    ratio_threshold: float = 0.65,
    max_generate_per_class: Optional[int] = None,
) -> Dict[int, int]:
    """
    Compute per-class synthetic image counts to reduce imbalance.

    Any class with count < ratio_threshold * majority_count is considered minority.
    Target count for minority classes is majority_count.
    """
    labels = df["diagnosis"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=5)
    majority = int(counts.max()) if len(counts) > 0 else 0

    plan: Dict[int, int] = {}
    for cls_id, count in enumerate(counts):
        if count <= 0:
            continue

        if count < majority * ratio_threshold:
            needed = max(0, majority - int(count))
            if max_generate_per_class is not None:
                needed = min(needed, int(max_generate_per_class))
            if needed > 0:
                plan[int(cls_id)] = int(needed)

    return plan


def save_augmented_dataframe(
    original_df: pd.DataFrame,
    generated_records: List[Tuple[str, int]],
    output_csv_path: Path,
) -> pd.DataFrame:
    synth_df = pd.DataFrame(generated_records, columns=["id_code", "diagnosis"])
    augmented_df = pd.concat([original_df[["id_code", "diagnosis"]], synth_df], ignore_index=True)
    augmented_df.to_csv(output_csv_path, index=False)
    return augmented_df


def run_cgan_augmentation(
    csv_path: Path,
    image_dir: Path,
    output_dir: Path,
    config: CGANConfig,
    ratio_threshold: float = 0.65,
    max_generate_per_class: Optional[int] = None,
    save_size: int = 224,
) -> Dict[str, object]:
    """Run full cGAN training + minority sample generation pipeline."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_dataframe(csv_path)
    plan = compute_generation_plan(
        df,
        ratio_threshold=ratio_threshold,
        max_generate_per_class=max_generate_per_class,
    )

    if not plan:
        return {
            "status": "no_minority_gap",
            "message": "Dataset is already balanced enough for the configured threshold.",
            "generation_plan": {},
        }

    dataset = APTOSCGANDataset(
        df=df,
        image_dir=image_dir,
        image_size=config.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    trainer = ConditionalGANTrainer(config)
    history = trainer.train(loader, output_dir=output_dir / "checkpoints")

    generated_records: List[Tuple[str, int]] = []
    synth_dir = image_dir
    for cls_id, amount in sorted(plan.items()):
        prefix = f"synthetic_{int(time.time())}"
        ids = trainer.generate_images(
            class_id=cls_id,
            num_images=amount,
            out_dir=synth_dir,
            prefix=prefix,
            save_size=save_size,
        )
        generated_records.extend((img_id, cls_id) for img_id in ids)

    augmented_csv_path = output_dir / "train_augmented.csv"
    augmented_df = save_augmented_dataframe(df, generated_records, augmented_csv_path)

    result = {
        "status": "ok",
        "base_samples": int(len(df)),
        "generated_samples": int(len(generated_records)),
        "augmented_samples": int(len(augmented_df)),
        "generation_plan": {str(k): int(v) for k, v in plan.items()},
        "history": {
            "g_loss_final": history["g_loss"][-1] if history["g_loss"] else None,
            "d_loss_final": history["d_loss"][-1] if history["d_loss"] else None,
        },
        "augmented_csv_path": str(augmented_csv_path),
        "elapsed_sec": float(time.time() - t0),
    }

    with open(output_dir / "cgan_augmentation_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
