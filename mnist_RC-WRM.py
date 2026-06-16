from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


VERSION = "mnist_RC-WRM"


@dataclass
class Config:
    outdir: str = "runs/mnist_RC-WRM"
    data_dir: str = "./data"
    seed: int = 12345
    device: str = "auto"
    rho0: float = 0.3
    fixed_gammas: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    gamma_init: float = 0.4
    gamma_min: float = 0.10
    gamma_max: float = 1.50
    eta_gamma: float = 0.15
    rho_ema_beta: float = 0.5
    phase_epochs: int = 1
    epochs: int = 4
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    label_smoothing: float = 0.1
    n_train: int = 12000
    n_calib: int = 2000
    n_test: int = 2000
    wrm_steps: int = 5
    eval_wrm_steps: int = 12
    wrm_step_scale: float = 1.0
    calibration_batches: int = 8
    pgd_steps: int = 20
    pgd_points: int = 15
    pgd_eval_batches: int = 10
    num_workers: int = 2
    gamma_update_samples: int = 2000
    gamma_update_batches: int = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        keys, seen = [], set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def append_csv(path: Path, row: Dict, fieldnames: List[str]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def fval(row: Dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def fixed_dir_name(gamma: float, epochs: int) -> str:
    return f"fixed_gamma_{gamma:g}_epochs_{epochs}"


def require_file(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")


def cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sq_l2_per_example(x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return (x - x0).view(x.shape[0], -1).pow(2).sum(dim=1)


def ce_vec(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)


class SmallELUCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.ELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ELU(), nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loaders(cfg: Config):
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=transform)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(train_full))
    n_train = min(cfg.n_train, len(train_full))
    n_calib = min(cfg.n_calib, len(train_full) - n_train)
    train_set = Subset(train_full, idx[:n_train].tolist())
    calib_set = Subset(train_full, idx[n_train:n_train + n_calib].tolist())
    test_idx = rng.permutation(len(test_full))[:min(cfg.n_test, len(test_full))]
    test_set = Subset(test_full, test_idx.tolist())
    kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_set, shuffle=True, **kwargs),
        DataLoader(calib_set, shuffle=False, **kwargs),
        DataLoader(test_set, shuffle=False, **kwargs),
    )


@torch.no_grad()
def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def wrm_attack(model, x, y, gamma: float, steps: int, step_scale: float, label_smoothing: float = 0.0):
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    eps = 0.5 / gamma
    x0 = x.detach()
    x_adv = x0.clone().detach().requires_grad_(True)
    grad0 = torch.autograd.grad((eps * ce_vec(model(x_adv), y, label_smoothing)).sum(), x_adv)[0]
    x_adv = clamp01(x0 + grad0.detach()).detach()
    for t in range(steps):
        x_adv.requires_grad_(True)
        loss_vec = ce_vec(model(x_adv), y, label_smoothing)
        cost_vec = sq_l2_per_example(x_adv, x0)
        obj = (eps * loss_vec - 0.5 * cost_vec).sum()
        grad = torch.autograd.grad(obj, x_adv)[0]
        x_adv = clamp01(x_adv + (step_scale / math.sqrt(t + 2.0)) * grad).detach()
    return x_adv


def train_epoch(model, loader, optim, dev, gamma: float, cfg: Config):
    model.train()
    n = 0
    loss_sum = 0.0
    acc = 0
    rho_sum = 0.0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        x_adv = wrm_attack(model, x, y, gamma, cfg.wrm_steps, cfg.wrm_step_scale, cfg.label_smoothing)
        with torch.no_grad():
            rho_sum += sq_l2_per_example(x_adv, x).sum().item()
        optim.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
        loss.backward()
        optim.step()
        with torch.no_grad():
            n += x.shape[0]
            loss_sum += loss.item() * x.shape[0]
            acc += (logits.argmax(1) == y).sum().item()
    return {"train_loss": loss_sum / n, "train_acc_on_adv_inputs": acc / n, "train_rho_batch": rho_sum / n}


@torch.no_grad()
def eval_clean(model, loader, dev, max_batches: Optional[int] = None):
    model.eval()
    n = 0
    correct = 0
    loss_sum = 0.0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += x.shape[0]
    return {"clean_acc": correct / n, "clean_loss": loss_sum / n}


def eval_wrm(model, loader, dev, gamma: float, cfg: Config, max_batches: Optional[int]):
    model.eval()
    n = 0
    correct = 0
    rho_sum = 0.0
    loss_sum = 0.0
    phi_sum = 0.0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        x_adv = wrm_attack(model, x, y, gamma, cfg.eval_wrm_steps, cfg.wrm_step_scale, 0.0)
        with torch.no_grad():
            logits = model(x_adv)
            loss_vec = ce_vec(logits, y, 0.0)
            cost_vec = sq_l2_per_example(x_adv, x)
            n += x.shape[0]
            correct += (logits.argmax(1) == y).sum().item()
            rho_sum += cost_vec.sum().item()
            loss_sum += loss_vec.sum().item()
            phi_sum += (loss_vec - gamma * cost_vec).sum().item()
    rho = rho_sum / n
    return {"rho": rho, "sqrt_rho": math.sqrt(max(rho, 0.0)), "wrm_acc": correct / n, "wrm_loss": loss_sum / n, "phi": phi_sum / n}


def load_model_from_dir(run_dir: Path, device: torch.device):
    ckpt_path = run_dir / "model.pt"
    require_file(ckpt_path, "model checkpoint")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SmallELUCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def make_update_steps(total_model_updates: int, num_segments: int) -> List[int]:
    if num_segments < 2:
        raise ValueError("num_segments must be at least 2")
    steps = []
    for i in range(1, num_segments):
        step = int(round(i * total_model_updates / float(num_segments)))
        step = max(1, min(step, total_model_updates - 1))
        steps.append(step)
    seen = set()
    unique = []
    for s in steps:
        if s not in seen:
            unique.append(s)
            seen.add(s)
    return unique


def make_train_gamma_loader(train_loader, cfg, n_gamma_update_samples: int, seed: int):
    train_dataset = train_loader.dataset
    n = min(int(n_gamma_update_samples), len(train_dataset))
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(train_dataset))[:n].tolist()
    subset = Subset(train_dataset, idx)
    kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return DataLoader(subset, **kwargs)


def update_gamma(gamma: float, rho_ema: float, cfg) -> float:
    return float(min(max(gamma + cfg.eta_gamma * (rho_ema - cfg.rho0), cfg.gamma_min), cfg.gamma_max))


def summarize_history(run_dir: Path, label: str, method: str, rho0: float, last_k: int) -> Dict:
    hist_path = run_dir / "history.csv"
    require_file(hist_path, "history")
    rows = read_csv(hist_path)
    final = rows[-1]
    tail = rows[-min(last_k, len(rows)):]
    rho_tail = np.asarray([fval(r, "rho_calib") for r in tail], dtype=float)
    return {
        "method": method,
        "label": label,
        "outdir": str(run_dir),
        "final_epoch": int(fval(final, "epoch", 0)),
        "final_gamma": fval(final, "gamma_after"),
        "final_rho": fval(final, "rho_calib"),
        "last_mean_rho": float(np.nanmean(rho_tail)),
        "last_std_rho": float(np.nanstd(rho_tail)),
        "last_mean_abs_error_to_target": float(np.nanmean(np.abs(rho_tail - rho0))),
        "target_rho0": rho0,
    }


def train_fixed_timed(cfg, loaders, device: torch.device, run_dir: Path, gamma: float) -> Dict:
    train_loader, calib_loader, test_loader = loaders
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_path = run_dir / "history.csv"
    if hist_path.exists():
        hist_path.unlink()
    model = SmallELUCNN().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    fields = [
        "epoch", "method", "gamma_before", "gamma_after", "rho0",
        "rho_calib", "rho_calib_ema", "radius_error", "abs_radius_error",
        "calib_wrm_acc", "test_clean_acc", "test_clean_loss", "test_wrm_acc",
        "test_rho", "test_phi", "train_loss", "train_acc_on_adv_inputs",
        "train_rho_batch", "epoch_seconds", "train_seconds", "eval_seconds",
    ]
    total_train_seconds = 0.0
    total_eval_seconds = 0.0
    cuda_sync_if_needed(device)
    total_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        cuda_sync_if_needed(device)
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        tr = train_epoch(model, train_loader, optim, device, float(gamma), cfg)
        cuda_sync_if_needed(device)
        train_seconds = time.perf_counter() - train_start
        eval_start = time.perf_counter()
        calib = eval_wrm(model, calib_loader, device, float(gamma), cfg, cfg.calibration_batches)
        clean = eval_clean(model, test_loader, device, cfg.calibration_batches)
        wrm_t = eval_wrm(model, test_loader, device, float(gamma), cfg, cfg.calibration_batches)
        cuda_sync_if_needed(device)
        eval_seconds = time.perf_counter() - eval_start
        total_train_seconds += train_seconds
        total_eval_seconds += eval_seconds
        row = {
            "epoch": epoch,
            "method": "fixed",
            "gamma_before": float(gamma),
            "gamma_after": float(gamma),
            "rho0": cfg.rho0,
            "rho_calib": calib["rho"],
            "rho_calib_ema": calib["rho"],
            "radius_error": calib["rho"] - cfg.rho0,
            "abs_radius_error": abs(calib["rho"] - cfg.rho0),
            "calib_wrm_acc": calib["wrm_acc"],
            "test_clean_acc": clean["clean_acc"],
            "test_clean_loss": clean["clean_loss"],
            "test_wrm_acc": wrm_t["wrm_acc"],
            "test_rho": wrm_t["rho"],
            "test_phi": wrm_t["phi"],
            "epoch_seconds": time.perf_counter() - epoch_start,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            **tr,
        }
        append_csv(hist_path, row, fields)
    cuda_sync_if_needed(device)
    total_seconds = time.perf_counter() - total_start
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "method": "fixed",
        "final_gamma": float(gamma),
        "version": VERSION,
    }, run_dir / "model.pt")
    return {
        "method": f"fixed_gamma_{gamma:g}",
        "label": rf"Fixed WRM, $\gamma={gamma:g}$",
        "outdir": str(run_dir),
        "total_seconds": total_seconds,
        "training_seconds": total_train_seconds,
        "evaluation_seconds": total_eval_seconds,
        "gamma_update_seconds": 0.0,
        "gamma_update_count": 0,
    }


def train_radius_controlled_timed(cfg, loaders, gamma_update_loader, device: torch.device, run_dir: Path, gamma_init: float, num_segments: int) -> Dict:
    train_loader, calib_loader, test_loader = loaders
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_path = run_dir / "history.csv"
    update_path = run_dir / "gamma_updates.csv"
    for p in [hist_path, update_path]:
        if p.exists():
            p.unlink()
    model = SmallELUCNN().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    batches_per_epoch = len(train_loader)
    total_model_updates = int(cfg.epochs) * batches_per_epoch
    update_steps = set(make_update_steps(total_model_updates, num_segments))
    hist_fields = [
        "epoch", "method", "gamma_before", "gamma_after", "rho0",
        "rho_calib", "rho_calib_ema", "radius_error", "abs_radius_error",
        "calib_wrm_acc", "test_clean_acc", "test_clean_loss", "test_wrm_acc",
        "test_rho", "test_phi", "train_loss", "train_acc_on_adv_inputs",
        "train_rho_batch", "global_step", "gamma_update_count",
        "epoch_seconds", "train_seconds", "eval_seconds", "gamma_update_seconds_epoch",
    ]
    update_fields = [
        "gamma_update_index", "global_step", "epoch", "batch_in_epoch",
        "gamma_before", "gamma_after", "rho_train_update", "rho_ema",
        "rho0", "radius_error_update", "abs_radius_error_update",
        "gamma_update_seconds",
    ]
    cur_gamma = float(gamma_init)
    rho_ema = None
    global_step = 0
    gamma_update_count = 0
    total_train_seconds = 0.0
    total_eval_seconds = 0.0
    total_gamma_update_seconds = 0.0
    cuda_sync_if_needed(device)
    total_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        gamma_epoch_start = cur_gamma
        n = 0
        loss_sum = 0.0
        acc_sum = 0
        rho_sum = 0.0
        gamma_update_seconds_epoch = 0.0
        cuda_sync_if_needed(device)
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_adv = wrm_attack(
                model,
                x,
                y,
                gamma=cur_gamma,
                steps=cfg.wrm_steps,
                step_scale=cfg.wrm_step_scale,
                label_smoothing=cfg.label_smoothing,
            )
            with torch.no_grad():
                rho_batch = float(sq_l2_per_example(x_adv, x).mean().item())
                rho_sum += rho_batch * x.shape[0]
            optim.zero_grad(set_to_none=True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
            loss.backward()
            optim.step()
            with torch.no_grad():
                n += x.shape[0]
                loss_sum += loss.item() * x.shape[0]
                acc_sum += (logits.argmax(1) == y).sum().item()
            global_step += 1
            if global_step in update_steps:
                cuda_sync_if_needed(device)
                update_start = time.perf_counter()
                update_metrics = eval_wrm(
                    model,
                    gamma_update_loader,
                    device,
                    gamma=cur_gamma,
                    cfg=cfg,
                    max_batches=cfg.gamma_update_batches,
                )
                cuda_sync_if_needed(device)
                update_seconds = time.perf_counter() - update_start
                gamma_update_seconds_epoch += update_seconds
                total_gamma_update_seconds += update_seconds
                rho_update = float(update_metrics["rho"])
                rho_ema = rho_update if rho_ema is None else cfg.rho_ema_beta * rho_ema + (1.0 - cfg.rho_ema_beta) * rho_update
                gamma_before = cur_gamma
                cur_gamma = update_gamma(cur_gamma, rho_ema, cfg)
                gamma_update_count += 1
                append_csv(update_path, {
                    "gamma_update_index": gamma_update_count,
                    "global_step": global_step,
                    "epoch": epoch,
                    "batch_in_epoch": batch_idx,
                    "gamma_before": gamma_before,
                    "gamma_after": cur_gamma,
                    "rho_train_update": rho_update,
                    "rho_ema": rho_ema,
                    "rho0": cfg.rho0,
                    "radius_error_update": rho_update - cfg.rho0,
                    "abs_radius_error_update": abs(rho_update - cfg.rho0),
                    "gamma_update_seconds": update_seconds,
                }, update_fields)
        cuda_sync_if_needed(device)
        train_seconds = time.perf_counter() - train_start
        total_train_seconds += train_seconds
        eval_start = time.perf_counter()
        calib = eval_wrm(model, calib_loader, device, cur_gamma, cfg, cfg.calibration_batches)
        clean = eval_clean(model, test_loader, device, cfg.calibration_batches)
        wrm_t = eval_wrm(model, test_loader, device, cur_gamma, cfg, cfg.calibration_batches)
        cuda_sync_if_needed(device)
        eval_seconds = time.perf_counter() - eval_start
        total_eval_seconds += eval_seconds
        row = {
            "epoch": epoch,
            "method": "radius_controlled",
            "gamma_before": gamma_epoch_start,
            "gamma_after": cur_gamma,
            "rho0": cfg.rho0,
            "rho_calib": calib["rho"],
            "rho_calib_ema": rho_ema if rho_ema is not None else calib["rho"],
            "radius_error": calib["rho"] - cfg.rho0,
            "abs_radius_error": abs(calib["rho"] - cfg.rho0),
            "calib_wrm_acc": calib["wrm_acc"],
            "test_clean_acc": clean["clean_acc"],
            "test_clean_loss": clean["clean_loss"],
            "test_wrm_acc": wrm_t["wrm_acc"],
            "test_rho": wrm_t["rho"],
            "test_phi": wrm_t["phi"],
            "train_loss": loss_sum / n,
            "train_acc_on_adv_inputs": acc_sum / n,
            "train_rho_batch": rho_sum / n,
            "global_step": global_step,
            "gamma_update_count": gamma_update_count,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "gamma_update_seconds_epoch": gamma_update_seconds_epoch,
        }
        append_csv(hist_path, row, hist_fields)
    cuda_sync_if_needed(device)
    total_seconds = time.perf_counter() - total_start
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "method": "radius_controlled",
        "final_gamma": cur_gamma,
        "gamma_update_count": gamma_update_count,
        "version": VERSION,
    }, run_dir / "model.pt")
    return {
        "method": "radius_controlled",
        "label": "Radius-Controlled WRM",
        "outdir": str(run_dir),
        "total_seconds": total_seconds,
        "training_seconds": total_train_seconds,
        "evaluation_seconds": total_eval_seconds,
        "gamma_update_seconds": total_gamma_update_seconds,
        "gamma_update_count": gamma_update_count,
    }


def compute_phi_grid(model, test_loader, device, cfg, gamma_adv_grid: List[float], eval_batches: Optional[int], label: str, method: str, source_outdir: str):
    rows = []
    for gamma_adv in gamma_adv_grid:
        metrics = eval_wrm(
            model,
            test_loader,
            device,
            gamma=float(gamma_adv),
            cfg=cfg,
            max_batches=eval_batches,
        )
        phi = float(metrics["phi"])
        rho_hat = float(metrics["rho"])
        wrm_loss = float(metrics["wrm_loss"])
        rows.append({
            "method": method,
            "label": label,
            "source_outdir": source_outdir,
            "gamma_adv": float(gamma_adv),
            "phi": phi,
            "rho_hat": rho_hat,
            "sqrt_rho_hat": math.sqrt(max(rho_hat, 0.0)),
            "transported_loss": wrm_loss,
            "phi_plus_gamma_rho_hat": phi + float(gamma_adv) * rho_hat,
            "wrm_acc": float(metrics["wrm_acc"]),
        })
    return rows


def make_dual_curve(phi_rows: List[Dict], rho_grid: np.ndarray):
    by_method: Dict[str, List[Dict]] = {}
    for row in phi_rows:
        by_method.setdefault(row["method"], []).append(row)
    curve_rows = []
    for method, rows in by_method.items():
        label = rows[0]["label"]
        for rho in rho_grid:
            vals = []
            for r in rows:
                gamma = float(r["gamma_adv"])
                val = float(r["phi"]) + gamma * float(rho)
                vals.append((val, gamma, float(r["phi"]), float(r["rho_hat"]), float(r["transported_loss"])))
            best_val, best_gamma, best_phi, rho_hat_at_best, transported_loss_at_best = min(vals, key=lambda t: t[0])
            curve_rows.append({
                "method": method,
                "label": label,
                "rho": float(rho),
                "approx_worstcase_loss": float(best_val),
                "best_gamma_adv": float(best_gamma),
                "best_phi": float(best_phi),
                "rho_hat_at_best_gamma": float(rho_hat_at_best),
                "transported_loss_at_best_gamma": float(transported_loss_at_best),
            })
    return curve_rows


def save_effective_radius_plot(outdir: Path, summaries: List[Dict], rho0: float) -> None:
    fixed = [s for s in summaries if str(s["method"]).startswith("fixed_gamma_")]
    closest = sorted(fixed, key=lambda s: abs(float(s["last_mean_rho"]) - rho0))[:2]
    selected = [s for s in summaries if s["method"] == "radius_controlled"] + closest
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "legend.fontsize": 10, "xtick.labelsize": 13, "ytick.labelsize": 13})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for s in selected:
        rows = read_csv(Path(s["outdir"]) / "history.csv")
        ax.plot(
            [fval(r, "epoch") for r in rows],
            [fval(r, "rho_calib") for r in rows],
            marker="o",
            markersize=3,
            linewidth=2.2,
            label=s["label"],
        )
    ax.axhline(rho0, linestyle="--", linewidth=2.0, label=rf"target $\rho_0 = {rho0:g}$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Effective radius $\widehat{\rho}$")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outdir / "effective_radius.png", dpi=300)
    plt.close(fig)


def save_worstcase_plot(outdir: Path, curve_rows: List[Dict], rho0: float) -> None:
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "legend.fontsize": 10, "xtick.labelsize": 13, "ytick.labelsize": 13})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in dict.fromkeys(r["method"] for r in curve_rows):
        rows = sorted([r for r in curve_rows if r["method"] == method], key=lambda r: float(r["rho"]))
        ax.plot(
            [float(r["rho"]) for r in rows],
            [float(r["approx_worstcase_loss"]) for r in rows],
            marker="o",
            markersize=2.5,
            linewidth=2.0,
            label=rows[0]["label"],
        )
    ax.axvline(rho0, linestyle="--", linewidth=2.0, label=rf"target $\rho_0 = {rho0:g}$")
    ax.set_xlabel(r"Wasserstein radius $\rho$")
    ax.set_ylabel("Wasserstein worst case loss")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outdir / "wasserstein_worstcase_loss.png", dpi=300)
    plt.close(fig)


def save_wall_clock(outdir: Path, timing_rows: List[Dict], dual_seconds: float, total_seconds: float) -> None:
    fixed = [r for r in timing_rows if str(r["method"]).startswith("fixed_gamma_")]
    rc = [r for r in timing_rows if r["method"] == "radius_controlled"][0]
    fixed_mean = float(np.mean([float(r["total_seconds"]) for r in fixed]))
    fixed_total = float(np.sum([float(r["total_seconds"]) for r in fixed]))
    rc_seconds = float(rc["total_seconds"])
    rows = []
    for r in timing_rows:
        rows.append({
            "method": r["method"],
            "label": r["label"],
            "total_seconds": r["total_seconds"],
            "training_seconds": r.get("training_seconds", ""),
            "evaluation_seconds": r.get("evaluation_seconds", ""),
            "gamma_update_seconds": r.get("gamma_update_seconds", ""),
            "gamma_update_count": r.get("gamma_update_count", ""),
        })
    rows.extend([
        {"method": "summary", "label": "average single fixed WRM", "total_seconds": fixed_mean},
        {"method": "summary", "label": "fixed WRM sweep total", "total_seconds": fixed_total},
        {"method": "summary", "label": "Radius-Controlled overhead vs average fixed (%)", "total_seconds": 100.0 * (rc_seconds / fixed_mean - 1.0)},
        {"method": "summary", "label": "Radius-Controlled cost relative to fixed sweep (%)", "total_seconds": 100.0 * rc_seconds / fixed_total},
        {"method": "summary", "label": "dual curve evaluation seconds", "total_seconds": dual_seconds},
        {"method": "summary", "label": "total script seconds", "total_seconds": total_seconds},
    ])
    write_csv(outdir / "wall_clock.csv", rows)
    lines = [
        f"Radius-Controlled WRM time: {rc_seconds:.3f} seconds",
        f"Average single fixed-WRM time: {fixed_mean:.3f} seconds",
        f"Fixed-WRM sweep total time: {fixed_total:.3f} seconds",
        f"Overhead vs average single fixed-WRM run: {100.0 * (rc_seconds / fixed_mean - 1.0):.2f}%",
        f"Cost relative to full fixed-WRM sweep: {100.0 * rc_seconds / fixed_total:.2f}%",
        f"Dual-curve evaluation time: {dual_seconds:.3f} seconds",
        f"Total script time: {total_seconds:.3f} seconds",
    ]
    (outdir / "wall_clock.txt").write_text("\n".join(lines) + "\n")


def cleanup_outputs(outdir: Path) -> None:
    keep = {"effective_radius.png", "wasserstein_worstcase_loss.png", "wall_clock.txt", "wall_clock.csv"}
    for child in list(outdir.iterdir()):
        if child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--outdir", default="runs/mnist_rho03_single_file_seed12345")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rho0", type=float, default=0.3)
    parser.add_argument("--rho-max", type=float, default=0.7)
    parser.add_argument("--rho-points", type=int, default=31)
    parser.add_argument("--fixed-gammas", type=float, nargs="*", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--gamma-init", type=float, default=0.4)
    parser.add_argument("--num-segments", type=int, default=6)
    parser.add_argument("--gamma-min", type=float, default=0.10)
    parser.add_argument("--gamma-max", type=float, default=1.50)
    parser.add_argument("--eta-gamma", type=float, default=0.15)
    parser.add_argument("--rho-ema-beta", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--n-train", type=int, default=12000)
    parser.add_argument("--n-calib", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--wrm-steps", type=int, default=5)
    parser.add_argument("--eval-wrm-steps", type=int, default=12)
    parser.add_argument("--wrm-step-scale", type=float, default=1.0)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--gamma-update-samples", type=int, default=2000)
    parser.add_argument("--gamma-update-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gamma-adv-grid", type=float, nargs="*", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80, 1.00, 1.25, 1.50, 2.00])
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--last-k", type=int, default=2)
    args = parser.parse_args()
    if args.version:
        print(VERSION)
        return
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = device_from_arg(args.device)
    cfg = Config(
        outdir=str(outdir),
        data_dir=args.data_dir,
        seed=args.seed,
        device=args.device,
        rho0=args.rho0,
        fixed_gammas=tuple(args.fixed_gammas),
        gamma_init=args.gamma_init,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        eta_gamma=args.eta_gamma,
        rho_ema_beta=args.rho_ema_beta,
        phase_epochs=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        n_train=args.n_train,
        n_calib=args.n_calib,
        n_test=args.n_test,
        wrm_steps=args.wrm_steps,
        eval_wrm_steps=args.eval_wrm_steps,
        wrm_step_scale=args.wrm_step_scale,
        calibration_batches=args.calibration_batches,
        pgd_steps=20,
        pgd_points=15,
        pgd_eval_batches=10,
        num_workers=args.num_workers,
        gamma_update_samples=args.gamma_update_samples,
        gamma_update_batches=args.gamma_update_batches,
    )
    train_loader, _, test_loader = make_loaders(cfg)
    loaders = (train_loader, test_loader, test_loader)
    gamma_update_loader = make_train_gamma_loader(train_loader, cfg, args.gamma_update_samples, args.seed + 777)
    start = time.perf_counter()
    rc_dir = outdir / f"radius_controlled_rho{args.rho0:g}_init_{args.gamma_init:g}_segments_{args.num_segments}"
    timing_rows = [train_radius_controlled_timed(cfg, loaders, gamma_update_loader, device, rc_dir, args.gamma_init, args.num_segments)]
    fixed_dirs = []
    for gamma in args.fixed_gammas:
        fixed_dir = outdir / fixed_dir_name(float(gamma), cfg.epochs)
        timing_rows.append(train_fixed_timed(cfg, loaders, device, fixed_dir, float(gamma)))
        fixed_dirs.append((float(gamma), fixed_dir))
    summaries = [summarize_history(rc_dir, "Radius-Controlled WRM", "radius_controlled", args.rho0, args.last_k)]
    model_specs = [{"method": "radius_controlled", "label": "Radius-Controlled WRM", "run_dir": rc_dir}]
    for gamma, run_dir in fixed_dirs:
        label = rf"Fixed WRM, $\gamma={gamma:g}$"
        method = f"fixed_gamma_{gamma:g}"
        summaries.append(summarize_history(run_dir, label, method, args.rho0, args.last_k))
        model_specs.append({"method": method, "label": label, "run_dir": run_dir})
    save_effective_radius_plot(outdir, summaries, args.rho0)
    cuda_sync_if_needed(device)
    dual_start = time.perf_counter()
    phi_rows = []
    max_batches = None if args.eval_batches == 0 else args.eval_batches
    for spec in model_specs:
        model = load_model_from_dir(Path(spec["run_dir"]), device)
        phi_rows.extend(compute_phi_grid(
            model=model,
            test_loader=test_loader,
            device=device,
            cfg=cfg,
            gamma_adv_grid=list(args.gamma_adv_grid),
            eval_batches=max_batches,
            label=spec["label"],
            method=spec["method"],
            source_outdir=str(spec["run_dir"]),
        ))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rho_grid = np.linspace(0.0, args.rho_max, args.rho_points)
    curve_rows = make_dual_curve(phi_rows, rho_grid)
    save_worstcase_plot(outdir, curve_rows, args.rho0)
    cuda_sync_if_needed(device)
    dual_seconds = time.perf_counter() - dual_start
    total_seconds = time.perf_counter() - start
    save_wall_clock(outdir, timing_rows, dual_seconds, total_seconds)
    cleanup_outputs(outdir)


if __name__ == "__main__":
    main()
