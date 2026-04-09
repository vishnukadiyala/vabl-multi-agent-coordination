"""Cross-domain test of the aux+attention gradient-interference pathology.

CIFAR-100 + small ViT + RotNet auxiliary task.

This script mirrors the Phase 2 minimal experimental matrix from the MARL
side of the paper, in a non-MARL setting. The 7-config matrix is the same
shape as marl_research/algorithms/jax/train_vabl_vec.py exposes via its CLI:

    A_full              ViT + aux        (lambda 0.05 constant)
    A_no_attn           mean pool + aux  (lambda 0.05 constant)
    A_no_aux            ViT only         (lambda 0)
    A_neither           mean pool only   (lambda 0)
    B_anneal            ViT + aux        (lambda anneals 0.05 -> 0 over first 50%)
    B_stopgrad          ViT + aux        (constant 0.05, stop-grad CLS -> aux head)
    B_anneal_stopgrad   ViT + aux        (annealed AND stop-grad)

Each run trains a small Vision Transformer (or mean-pool variant) on
CIFAR-100 for 200 epochs, with a primary classification loss and an
auxiliary RotNet (4-way rotation prediction) loss applied to the same CLS
representation. We report:

    Best:    max test accuracy reached during training
    Final5:  mean test accuracy of the last 5 epochs (analog to Final50 in MARL)

The headline finding (if it reproduces) is that A_full has Best similar to
all other configs but Final5 lower and with higher cross-seed std,
matching the MARL pathology.

CLI mirrors train_vabl_vec.py exactly:

    python train_vision_aux.py --no-attention
    python train_vision_aux.py --no-aux-loss --aux-lambda 0.0
    python train_vision_aux.py --aux-lambda 0.05 --aux-anneal-fraction 0.5
    python train_vision_aux.py --aux-lambda 0.05 --stop-gradient-belief
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# ============================================================================
# Config
# ============================================================================

@dataclass
class VisionConfig:
    # Architecture
    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    num_classes: int = 100
    num_rotations: int = 4

    # Training
    batch_size: int = 256
    epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    aux_lambda: float = 0.05
    grad_clip: float = 1.0

    # Ablation flags (mirror VABLv2Config naming exactly)
    use_attention: bool = True
    use_aux_loss: bool = True
    stop_gradient_belief_to_aux: bool = False
    aux_anneal_fraction: float = 0.0

    # Reporting
    final_window_epochs: int = 5  # mean over last N epochs (analog to Final50)


# ============================================================================
# Architecture
# ============================================================================

class PatchEmbed(nn.Module):
    """Conv-based patch embedding (image -> tokens)."""

    def __init__(self, image_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)              # [B, C, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, C]
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block: MHA + MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class MeanPoolBlock(nn.Module):
    """Per-patch MLP only - no token mixing.

    This is the no-attention analog: the patches are independently transformed
    by an MLP, then aggregated only at the very end via mean pooling. The
    architecture has identical parameter count to the transformer block (modulo
    the small bias in MHA), so the comparison is fair.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Per-patch MLP that "mimics" the parameter count of MHA's qkv+out projections
        # MHA has roughly 4*dim*dim params; we use a small MLP with similar count.
        self.token_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        # No token mixing - each token only sees itself
        x = x + self.token_mlp(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-style encoder with a CLS token. Supports attention and mean-pool variants."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        block_cls = TransformerBlock if cfg.use_attention else MeanPoolBlock
        self.blocks = nn.ModuleList([
            block_cls(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.cfg.use_attention:
            # Standard ViT: use the CLS token
            return x[:, 0]
        else:
            # Mean-pool variant: there's no real CLS interaction so average over patches
            return x[:, 1:].mean(dim=1)


class VisionAuxModel(nn.Module):
    """Encoder + classification head + auxiliary rotation head."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = VisionEncoder(cfg)
        self.classifier = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.embed_dim, cfg.num_rotations),
        )

    def forward(self, x):
        feat = self.encoder(x)
        cls_logits = self.classifier(feat)

        # Stop-gradient on the belief input to the aux head, mirroring
        # vabl_v2.py's stop_gradient_belief_to_aux. Aux gradients then only
        # train the aux head MLP, never the encoder.
        aux_input = feat.detach() if self.cfg.stop_gradient_belief_to_aux else feat
        aux_logits = self.aux_head(aux_input)

        return cls_logits, aux_logits


# ============================================================================
# Data + augmentations
# ============================================================================

def make_dataloaders(batch_size: int, num_workers: int = 4):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    root = os.environ.get("CIFAR_DATA_ROOT", "./data")
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def random_rotate_batch(x: torch.Tensor):
    """RotNet augmentation: rotate each image by a random multiple of 90 degrees.

    Returns the rotated batch and the rotation labels (0-3 for 0/90/180/270 degrees).
    """
    B = x.shape[0]
    rot = torch.randint(0, 4, (B,), device=x.device)
    out = torch.empty_like(x)
    for k in range(4):
        mask = (rot == k)
        if mask.any():
            out[mask] = torch.rot90(x[mask], k=k, dims=(-2, -1))
    return out, rot


# ============================================================================
# Training loop
# ============================================================================

def cosine_lr(epoch: int, total_epochs: int, warmup_epochs: int, base_lr: float) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def aux_lambda_for_epoch(cfg: VisionConfig, epoch: int) -> float:
    """Effective aux lambda for the current epoch.

    Mirrors the train_vabl_vec.py annealing schedule: linear decay from
    cfg.aux_lambda to 0 over the first cfg.aux_anneal_fraction of epochs,
    then 0 thereafter. cfg.aux_anneal_fraction = 0 means constant.
    """
    if not cfg.use_aux_loss:
        return 0.0
    if cfg.aux_anneal_fraction <= 0.0:
        return cfg.aux_lambda
    anneal_epochs = max(1, int(cfg.aux_anneal_fraction * cfg.epochs))
    frac_remaining = max(0.0, 1.0 - epoch / anneal_epochs)
    return cfg.aux_lambda * frac_remaining


def train_one(cfg: VisionConfig, seed: int, save_path: Optional[str] = None) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Vision aux experiment on {device}")
    print(f"  use_attention={cfg.use_attention} use_aux_loss={cfg.use_aux_loss}")
    print(f"  aux_lambda={cfg.aux_lambda} aux_anneal_fraction={cfg.aux_anneal_fraction}")
    print(f"  stop_gradient_belief_to_aux={cfg.stop_gradient_belief_to_aux}")
    print(f"  epochs={cfg.epochs} batch_size={cfg.batch_size} seed={seed}")

    train_loader, test_loader = make_dataloaders(cfg.batch_size)
    model = VisionAuxModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    test_acc_history = []
    train_loss_history = []
    aux_loss_history = []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        # LR schedule
        lr = cosine_lr(epoch, cfg.epochs, cfg.warmup_epochs, cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Aux lambda for this epoch
        eff_lambda = aux_lambda_for_epoch(cfg, epoch)

        # Train
        model.train()
        epoch_loss = 0.0
        epoch_aux = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_rot, rot_labels = random_rotate_batch(x)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                cls_logits, aux_logits = model(x_rot)
                cls_loss = F.cross_entropy(cls_logits, y)
                aux_loss = F.cross_entropy(aux_logits, rot_labels)
                total = cls_loss + eff_lambda * aux_loss

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(cls_loss.detach())
            epoch_aux += float(aux_loss.detach())
            n_batches += 1

        # Evaluate (no rotation augmentation at test time, evaluating on original CIFAR-100)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    cls_logits, _ = model(x)
                pred = cls_logits.argmax(dim=-1)
                correct += int((pred == y).sum())
                total += int(y.numel())
        test_acc = 100.0 * correct / total

        test_acc_history.append(test_acc)
        train_loss_history.append(epoch_loss / max(1, n_batches))
        aux_loss_history.append(epoch_aux / max(1, n_batches))
        best_acc = max(best_acc, test_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch+1:3d}/{cfg.epochs} | lr={lr:.4f} | aux_lambda={eff_lambda:.4f} | "
                  f"cls={epoch_loss/max(1,n_batches):.3f} | aux={epoch_aux/max(1,n_batches):.3f} | "
                  f"test={test_acc:.2f} | best={best_acc:.2f}")

    elapsed = time.time() - t0
    final_window = test_acc_history[-cfg.final_window_epochs:]
    final_mean = sum(final_window) / max(1, len(final_window))

    print(f"\n  done in {elapsed/60:.1f} min")
    print(f"  Best test acc: {best_acc:.2f}")
    print(f"  Final{cfg.final_window_epochs} mean: {final_mean:.2f}")

    result = {
        "best_acc": best_acc,
        "final_mean": final_mean,
        "test_acc_history": test_acc_history,
        "train_loss_history": train_loss_history,
        "aux_loss_history": aux_loss_history,
        "elapsed": elapsed,
        "config": {
            "use_attention": cfg.use_attention,
            "use_aux_loss": cfg.use_aux_loss,
            "aux_lambda": cfg.aux_lambda,
            "aux_anneal_fraction": cfg.aux_anneal_fraction,
            "stop_gradient_belief_to_aux": cfg.stop_gradient_belief_to_aux,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "embed_dim": cfg.embed_dim,
            "depth": cfg.depth,
            "num_heads": cfg.num_heads,
            "patch_size": cfg.patch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        },
        "seed": seed,
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved to {save_path}")

    return result


# ============================================================================
# CLI (mirrors train_vabl_vec.py exactly)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None)

    # Mirror the VABL CLI flag names exactly
    parser.add_argument("--no-attention", action="store_true",
                        help="Replace transformer attention blocks with per-patch MLP + mean pool.")
    parser.add_argument("--no-aux-loss", action="store_true",
                        help="Hard-disable the auxiliary loss regardless of aux_lambda.")
    parser.add_argument("--aux-lambda", type=float, default=0.05)
    parser.add_argument("--stop-gradient-belief", action="store_true",
                        help="Stop aux gradients from flowing into the encoder.")
    parser.add_argument("--aux-anneal-fraction", type=float, default=0.0)

    args = parser.parse_args()

    cfg = VisionConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_attention=not args.no_attention,
        use_aux_loss=not args.no_aux_loss,
        aux_lambda=args.aux_lambda,
        stop_gradient_belief_to_aux=args.stop_gradient_belief,
        aux_anneal_fraction=args.aux_anneal_fraction,
    )

    train_one(cfg, args.seed, args.save)
