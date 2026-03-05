"""
Train RASEM (AFNO-based segmentation model)
"""

import os
import time
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tv
import cv2
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info

from config import (
    TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR,
    VAL_IMAGES_DIR,   VAL_MASKS_DIR,
    TEST_IMAGES_DIR,  TEST_MASKS_DIR,
    BATCH_SIZE, NUM_WORKERS, NUM_CLASSES, IMAGE_SIZE,
    LEARNING_RATE, NUM_EPOCHS, MEAN, STD, OUT_DIR
)

# Import your custom model file (save previous model code as rasem_model.py)
from rasem_afno import RASEM


# --------------------------------
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------- Dataset ----------------
class NopalDataset(Dataset):
    def __init__(self, img_dir, msk_dir, augment=False):
        self.imgs = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.msks = sorted([
            os.path.join(msk_dir, f) for f in os.listdir(msk_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        assert len(self.imgs) == len(self.msks) and len(self.imgs) > 0, "Dataset vacío o desparejo."
        self.augment = augment
        self.to_tensor = tv.ToTensor()
        self.norm = tv.Normalize(mean=MEAN, std=STD)

    def _geom_aug(self, img, msk):
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            msk = np.fliplr(msk).copy()
        if random.random() > 0.5:
            img = np.flipud(img).copy()
            msk = np.flipud(msk).copy()
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k).copy()
            msk = np.rot90(msk, k).copy()
        return img, msk

    def _photo_aug(self, img, p=0.3, b_lim=0.2, c_lim=0.2):
        if random.random() >= p:
            return img
        imgf = img.astype(np.float32)
        mean = imgf.mean(axis=(0, 1), keepdims=True)  # per-channel mean
        c = 1.0 + random.uniform(-c_lim, c_lim)       # contrast
        b = 1.0 + random.uniform(-b_lim, b_lim)       # brightness
        imgf = (imgf - mean) * c + mean
        imgf = imgf * b
        return np.clip(imgf, 0, 255).astype(np.uint8)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.msks[idx], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, msk = self._geom_aug(img, msk)
            img = self._photo_aug(img)

        img_raw = img.copy()
        msk_raw = ((msk > 127).astype(np.uint8) * 255)

        img_t = self.to_tensor(img)      # [0,1]
        img_t = self.norm(img_t)         # ImageNet normalization
        msk_t = torch.from_numpy((msk > 127).astype(np.uint8)).long()

        return {
            "image": img_t,
            "mask": msk_t,
            "image_raw": img_raw,
            "mask_raw": msk_raw
        }


# ---------------- helpers ----------------
def up_to_mask(p, t):
    return torch.nn.functional.interpolate(
        p, size=t.shape[-2:], mode='bilinear', align_corners=False
    )


@torch.no_grad()
def compute_metrics_binary(logits, masks, thr=0.5):
    # logits: [B,1,h,w]
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = up_to_mask(logits, masks)
    prob = torch.sigmoid(logits[:, 0])
    pred = (prob > thr).float()
    maskf = masks.float()

    acc = (pred == maskf).float().mean()
    rmse = torch.sqrt(((prob - maskf) ** 2).mean())
    inter = (pred * maskf).sum()
    union = pred.sum() + maskf.sum() - inter + 1e-6
    iou = inter / union
    return acc.item(), rmse.item(), iou.item()


@torch.no_grad()
def compute_metrics_multiclass(logits, masks):
    # logits: [B,C,h,w], masks: [B,h,w] in [0..C-1]
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = up_to_mask(logits, masks)
    pred = torch.argmax(logits, dim=1)
    acc = (pred == masks).float().mean()

    # mean IoU (macro over classes present in GT or pred)
    num_classes = logits.shape[1]
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (masks == c)
        inter = (pred_c & gt_c).sum().float()
        union = (pred_c | gt_c).sum().float()
        if union > 0:
            ious.append((inter / (union + 1e-6)).item())
    miou = float(np.mean(ious)) if len(ious) > 0 else 0.0

    # RMSE on class index map (optional, for continuity with your logs)
    rmse = torch.sqrt(((pred.float() - masks.float()) ** 2).mean())
    return acc.item(), rmse.item(), miou


def dice_loss_binary(p, t, eps=1e-6):
    if p.shape[-2:] != t.shape[-2:]:
        p = up_to_mask(p, t)
    prob = torch.sigmoid(p)
    t = t.float().unsqueeze(1)
    inter = (prob * t).sum((1, 2, 3))
    union = prob.sum((1, 2, 3)) + t.sum((1, 2, 3)) + eps
    return 1 - (2 * inter / union).mean()


def combined_loss_binary(p, t, bce):
    if p.shape[-2:] != t.shape[-2:]:
        p = up_to_mask(p, t)
    return bce(p, t.float().unsqueeze(1)) + 0.5 * dice_loss_binary(p, t)


def save_panel_binary(img_rgb, mask_gt, logits_1ch, out_path, thr=0.5):
    # logits_1ch: [h,w] torch
    H, W = mask_gt.shape
    logits_1ch = torch.nn.functional.interpolate(
        logits_1ch.unsqueeze(0).unsqueeze(0),  # [1,1,h,w]
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    prob = torch.sigmoid(logits_1ch).cpu().numpy()
    pred = (prob > thr).astype(np.uint8) * 255

    overlay = img_rgb.copy()
    overlay[pred == 255] = (255, 0, 0)

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(img_rgb); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(mask_gt, cmap="gray"); ax[1].set_title("GT"); ax[1].axis("off")
    ax[2].imshow(pred, cmap="gray"); ax[2].set_title("Pred"); ax[2].axis("off")
    ax[3].imshow(overlay); ax[3].set_title("Overlay"); ax[3].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------- train + test ----------------
def main():
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    os.makedirs(OUT_DIR, exist_ok=True)

    model = RASEM(
        num_classes=NUM_CLASSES,
        embed_dims=[64,128,320,512],
        in_channels=3,
        mlp_ratios=[8, 8, 6, 4],
        depths=[3, 5, 27, 3],
        drop_rate=0.0,
        drop_path_rate=0.1,
        afno_num_blocks=8,
        afno_hard_thresholding_fraction=0.4,
        afno_sparsity_threshold=0.005
    ).to(device)

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model, 
            (3, 224, 224), 
            as_strings=True,
            print_per_layer_stat=False
        )
        print(f"FLOPs: {macs}, Parameters: {params}")


    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Loss setup
    is_binary = (NUM_CLASSES == 1)
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Data
    tr_ds = NopalDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, augment=True)
    va_ds = NopalDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, augment=False)
    te_ds = NopalDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR, augment=False)

    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    hist = {k: [] for k in [
        "train_loss", "train_acc", "train_rmse", "train_iou",
        "val_loss", "val_acc", "val_rmse", "val_iou", "t_img"
    ]}
    best_iou = -1.0

    for ep in range(1, NUM_EPOCHS + 1):
        # ---- Train ----
        model.train()
        tl = ta = tr = ti = 0.0

        for batch in tr_ld:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(img)  # [B,C,h,w]

            if is_binary:
                loss = combined_loss_binary(logits, msk, criterion)
                a, r, i = compute_metrics_binary(logits, msk)
            else:
                if logits.shape[-2:] != msk.shape[-2:]:
                    logits = up_to_mask(logits, msk)
                loss = criterion(logits, msk)
                a, r, i = compute_metrics_multiclass(logits, msk)

            loss.backward()
            opt.step()

            tl += loss.item()
            ta += a
            tr += r
            ti += i

        ntr = len(tr_ld)
        tl, ta, tr, ti = [x / ntr for x in (tl, ta, tr, ti)]

        # ---- Validation ----
        model.eval()
        vl = va = vr = vi = 0.0
        t_sum = 0.0
        n_img = 0

        with torch.no_grad():
            for batch in va_ld:
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)

                t0 = time.perf_counter()
                logits = model(img)
                t_sum += (time.perf_counter() - t0)
                n_img += img.size(0)

                if is_binary:
                    l = combined_loss_binary(logits, msk, criterion).item()
                    a, r, i = compute_metrics_binary(logits, msk)
                else:
                    if logits.shape[-2:] != msk.shape[-2:]:
                        logits = up_to_mask(logits, msk)
                    l = criterion(logits, msk).item()
                    a, r, i = compute_metrics_multiclass(logits, msk)

                vl += l
                va += a
                vr += r
                vi += i

        nva = len(va_ld)
        vl, va, vr, vi = [x / nva for x in (vl, va, vr, vi)]
        t_img = t_sum / max(1, n_img)

        if vi > best_iou:
            best_iou = vi
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "modelo_best.pth"))

        for k, v in zip(hist.keys(), [tl, ta, tr, ti, vl, va, vr, vi, t_img]):
            hist[k].append(v)

        print(
            f"Ep {ep:03d}/{NUM_EPOCHS} | "
            f"Train L:{tl:.4f} Acc:{ta:.4f} IoU:{ti:.4f} || "
            f"Val L:{vl:.4f} Acc:{va:.4f} IoU:{vi:.4f}"
        )

    # Save summary
    with open(os.path.join(OUT_DIR, "metricas_resumen.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "media", "desv_std"])
        for k in hist.keys():
            vals = hist[k]
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            w.writerow([k, float(np.mean(vals)), std])

    # ---- TEST ----
    best_ckpt = os.path.join(OUT_DIR, "modelo_best.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    test_out_dir = os.path.join(OUT_DIR, "Test_RASEM")
    os.makedirs(test_out_dir, exist_ok=True)

    logs = {k: [] for k in ("loss", "acc", "rmse", "iou", "t_img")}
    with torch.no_grad():
        for batch in te_ld:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)

            t0 = time.perf_counter()
            logits = model(img)
            dt = time.perf_counter() - t0

            if is_binary:
                l = combined_loss_binary(logits, msk, criterion).item()
                a, r, i = compute_metrics_binary(logits, msk)
            else:
                if logits.shape[-2:] != msk.shape[-2:]:
                    logits = up_to_mask(logits, msk)
                l = criterion(logits, msk).item()
                a, r, i = compute_metrics_multiclass(logits, msk)

            logs["loss"].append(l)
            logs["acc"].append(a)
            logs["rmse"].append(r)
            logs["iou"].append(i)
            logs["t_img"].append(dt / img.size(0))

    with open(os.path.join(test_out_dir, "test_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "media", "desv_std"])
        for k in logs.keys():
            vals = logs[k]
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            w.writerow([k, float(np.mean(vals)), std])

    # Save some visualizations (binary only in this panel function)
    if is_binary:
        idxs = random.sample(range(len(te_ds)), k=min(3, len(te_ds)))
        for j, idx in enumerate(idxs):
            sample = te_ds[idx]
            x = sample["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(x).cpu().squeeze(0)[0]  # [h,w] channel 0
            save_panel_binary(
                sample["image_raw"],
                sample["mask_raw"],
                logit,
                os.path.join(test_out_dir, f"vis_{j}.png"),
                thr=0.5
            )


if __name__ == "__main__":
    main()