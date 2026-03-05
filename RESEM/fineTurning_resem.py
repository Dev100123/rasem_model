"""
FINE TUNING (RASEM)
Nopal → Maguey → Nopal+Maguey
Phases:
0) Pre-test: direct inference on Maguey.
1) Fine-tuning: freeze encoder, train decoder (30 epochs) on Maguey.
2) Consolidation: unfreeze all, train on Mixed (Nopal+Maguey).
"""

import os, time, random, math, argparse, csv
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from contextlib import nullcontext

from config import (
    # Nopal
    TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR,
    VAL_IMAGES_DIR,   VAL_MASKS_DIR,
    TEST_IMAGES_DIR,  TEST_MASKS_DIR,
    # Maguey
    TRAIN_IMAGES_DIRM, TRAIN_MASKS_DIRM,
    VAL_IMAGES_DIRM,   VAL_MASKS_DIRM,
    TEST_IMAGES_DIRM,  TEST_MASKS_DIRM,
    # Hyperparams
    BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE, LEARNING_RATE, MEAN, STD
)

# Your custom model (file created before)
from RESEM.rasem_afno import RASEM


# ---------- Output dirs ----------
DIR_PRETEST = "Test_Maguey_RASEM"
DIR_FT      = "Test_Fine_Tuning_RASEM"
DIR_CONS    = "Test_Nopal_Maguey_RASEM"
for d in (DIR_PRETEST, DIR_FT, DIR_CONS):
    os.makedirs(d, exist_ok=True)

# ---------- Schedule ----------
FT_EPOCHS, CONS_EPOCHS = 30, 10
LR_FT, LR_CONS = LEARNING_RATE, 1e-5
PESOS_FT_BEST, PESOS_CONS_BEST = "rasem_ft_best.pth", "rasem_cons_best.pth"
PESOS_FT_LAST, PESOS_CONS_LAST = "rasem_ft_last.pth", "rasem_cons_last.pth"


# ---------- Repro ----------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)


# ---------- Dataset ----------
class SegDataset(Dataset):
    def __init__(self, img_dir, msk_dir, transform=None):
        self.imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.msks = sorted([os.path.join(msk_dir, f) for f in os.listdir(msk_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        assert len(self.imgs) == len(self.msks) and len(self.imgs) > 0, f"Empty/mismatched dataset: {img_dir}"
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.imgs[i]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.msks[i], cv2.IMREAD_GRAYSCALE)
        sample = {"image": img, "mask": msk}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class ToTensorResizeNorm:
    mean = np.array(MEAN, dtype=np.float32)
    std  = np.array(STD,  dtype=np.float32)

    def __call__(self, sample):
        img = sample["image"]
        msk = sample["mask"]

        # IMAGE_SIZE in your config is (H, W), cv2 expects (W, H)
        img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        msk = (msk > 0).astype(np.float32)  # binary

        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "mask": torch.tensor(msk, dtype=torch.float32)
        }


class RandomFlipRotate(ToTensorResizeNorm):
    def __call__(self, sample):
        img, msk = sample["image"], sample["mask"]
        if random.random() > 0.5:
            img, msk = np.fliplr(img), np.fliplr(msk)
        if random.random() > 0.5:
            img, msk = np.flipud(img), np.flipud(msk)
        k = random.choice([0, 1, 2])  # 0,90,180
        if k:
            img, msk = np.rot90(img, k), np.rot90(msk, k)
        return super().__call__({"image": img.copy(), "mask": msk.copy()})


train_tf = RandomFlipRotate()
test_tf  = ToTensorResizeNorm()


# ---------- Metrics / helpers ----------
def _iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum() + 1e-6
    return inter / union

def _rmse(pred, gt):
    return math.sqrt(((pred.astype(float) - gt.astype(float)) ** 2).mean())


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True, thr=0.5):
    model.eval()
    IoU, RM, AC, Ti = [], [], [], []

    amp_enabled = (use_amp and device.type == "cuda")
    autocast_ctx = (lambda: torch.amp.autocast("cuda")) if amp_enabled else nullcontext

    for b in loader:
        x = b["image"].to(device, non_blocking=True)  # (B,3,H,W)
        gt = (b["mask"].cpu().numpy() > 0)            # (B,H,W)

        t0 = time.perf_counter()
        with autocast_ctx():
            out = model(x)                            # (B,1,h,w)
        if out.shape[-2:] != b["mask"].shape[-2:]:
            out = torch.nn.functional.interpolate(out, size=b["mask"].shape[-2:], mode="bilinear", align_corners=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        Ti.append((time.perf_counter() - t0) / x.size(0))

        pr = (torch.sigmoid(out[:, 0]) > thr).cpu().numpy()  # (B,H,W)
        for i in range(pr.shape[0]):
            IoU.append(_iou(pr[i], gt[i]))
            RM.append(_rmse(pr[i], gt[i]))
            AC.append(np.mean(pr[i] == gt[i]))

    return {
        "IoU_mean": np.mean(IoU),  "IoU_std": np.std(IoU, ddof=1) if len(IoU) > 1 else 0.0,
        "RMSE_mean": np.mean(RM),  "RMSE_std": np.std(RM, ddof=1) if len(RM) > 1 else 0.0,
        "Acc_mean": np.mean(AC),   "Acc_std": np.std(AC, ddof=1) if len(AC) > 1 else 0.0,
        "t_img_mean": np.mean(Ti), "t_img_std": np.std(Ti, ddof=1) if len(Ti) > 1 else 0.0,
    }


def save_csv(stats, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "media", "desv_std"])
        for m in ["IoU", "RMSE", "Acc", "t_img"]:
            w.writerow([m, float(stats[f"{m}_mean"]), float(stats[f"{m}_std"])])


@torch.no_grad()
def save_examples(model, ds, device, out_dir, prefix, n=5, thr=0.5, use_amp=True):
    os.makedirs(out_dir, exist_ok=True)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model.eval()

    amp_enabled = (use_amp and device.type == "cuda")
    autocast_ctx = (lambda: torch.amp.autocast("cuda")) if amp_enabled else nullcontext

    for i, b in enumerate(ld):
        if i >= n:
            break
        x = b["image"].to(device)
        gt = (b["mask"].cpu().numpy() > 0)[0]

        with autocast_ctx():
            out = model(x)
        if out.shape[-2:] != b["mask"].shape[-2:]:
            out = torch.nn.functional.interpolate(out, size=b["mask"].shape[-2:], mode="bilinear", align_corners=False)

        pr = (torch.sigmoid(out[:, 0]) > thr).cpu().numpy()[0]

        # denorm image for visualization
        img_n = x.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        mean = np.array(MEAN, dtype=np.float32)
        std = np.array(STD, dtype=np.float32)
        img = np.clip((img_n * std + mean), 0, 1)
        img = (img * 255).astype(np.uint8)

        gt_v = (gt.astype(np.uint8) * 255)
        pr_v = (pr.astype(np.uint8) * 255)

        ov = img.copy()
        ov[pr] = (0.5 * np.array([255, 0, 0]) + 0.5 * ov[pr]).astype(np.uint8)

        panel = np.hstack([
            img,
            cv2.cvtColor(gt_v, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(pr_v, cv2.COLOR_GRAY2BGR),
            ov
        ])
        cv2.imwrite(os.path.join(out_dir, f"{prefix}{i}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def bce_dice_loss(logits, target):
    # logits: (B,1,H,W), target: (B,H,W) in {0,1}
    if target.dtype not in (torch.float32, torch.float64):
        target = target.float()
    if target.max() > 1.0:
        target = (target > 0).float()

    bce = nn.BCEWithLogitsLoss()(logits[:, 0], target)

    prob = torch.sigmoid(logits[:, 0])
    inter = (prob * target).sum(dim=(1, 2))
    union = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1e-6
    dice = 1.0 - (2.0 * inter / union).mean()
    return bce + 0.5 * dice


# ---------- Freeze/unfreeze for RASEM ----------
def freeze_encoder_rasem(model, freeze=True):
    # Freeze ONLY encoder for phase 1
    for p in model.encoder.parameters():
        p.requires_grad = (not freeze)
    # Keep decode head trainable
    for p in model.decode_head.parameters():
        p.requires_grad = True


def find_best_threshold(model, loader, device, use_amp=True):
    model.eval()
    thrs = np.linspace(0.30, 0.70, 9)
    scores = np.zeros_like(thrs, dtype=np.float64)

    amp_enabled = (use_amp and device.type == "cuda")
    autocast_ctx = (lambda: torch.amp.autocast("cuda")) if amp_enabled else nullcontext

    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device, non_blocking=True)
            y = (b["mask"].cpu().numpy() > 0).astype(np.uint8)  # (B,H,W)

            with autocast_ctx():
                out = model(x)  # (B,1,h,w)

            if out.shape[-2:] != b["mask"].shape[-2:]:
                out = torch.nn.functional.interpolate(out, size=b["mask"].shape[-2:], mode="bilinear", align_corners=False)

            prob = torch.sigmoid(out[:, 0]).cpu().numpy()

            for i, t in enumerate(thrs):
                pr = (prob > t).astype(np.uint8)
                inter = (pr & y).sum()
                uni = pr.sum() + y.sum() - inter + 1e-6
                scores[i] += inter / uni

    return float(thrs[np.argmax(scores)])


def eval_one_epoch(model, loader, device, thr=0.5):
    return evaluate(model, loader, device, thr=thr)["IoU_mean"]


def train_loop(model, loader_tr, loader_va, opt, epochs, device, ckpt_path, use_amp=True):
    best_iou = -1.0
    amp_enabled = (use_amp and device.type == "cuda")
    autocast_ctx = (lambda: torch.amp.autocast("cuda")) if amp_enabled else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for e in range(1, epochs + 1):
        model.train()
        running = 0.0

        for b in loader_tr:
            x = b["image"].to(device, non_blocking=True)
            y = b["mask"].to(device, non_blocking=True)

            if y.dtype not in (torch.float32, torch.float64):
                y = y.float()
            if y.max() > 1.0:
                y = (y > 0).float()

            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                out = model(x)
                if out.shape[-2:] != y.shape[-2:]:
                    out = torch.nn.functional.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = bce_dice_loss(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()

        iou_val = eval_one_epoch(model, loader_va, device, thr=0.5)
        print(f"Epoch {e:02d}/{epochs} | Loss {running/len(loader_tr):.4f} | IoU_val {iou_val:.4f}")

        if iou_val > best_iou:
            best_iou = iou_val
            torch.save(model.state_dict(), ckpt_path)


# ---------- Main ----------
def main(init_weights):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    model = RASEM(
        in_channels=3,
        num_classes=1,  # binary
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        depths=[3, 5, 27, 3],
        drop_rate=0.0,
        drop_path_rate=0.1,
        afno_num_blocks=8,
        afno_hard_thresholding_fraction=1.0,
        afno_sparsity_threshold=0.0
    ).to(dev)

    # Load Nopal-pretrained weights if provided
    if init_weights and os.path.isfile(init_weights):
        state = torch.load(init_weights, map_location=dev)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights: {init_weights}")
    else:
        print("Warning: initial weights not found, starting from scratch.")

    # ----- datasets/loaders -----
    ds_test_mag = SegDataset(TEST_IMAGES_DIRM, TEST_MASKS_DIRM, transform=test_tf)
    ld_test_mag = DataLoader(ds_test_mag, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 0) Pre-test on maguey
    st_pre = evaluate(model, ld_test_mag, dev, thr=0.5)
    save_csv(st_pre, DIR_PRETEST, "metricas_pre.csv")
    save_examples(model, ds_test_mag, dev, DIR_PRETEST, "mag_", 5, thr=0.5)

    # 1) Fine-tuning on maguey (freeze encoder)
    print("\nFine-tuning (Maguey)...")
    freeze_encoder_rasem(model, freeze=True)

    ds_mag_tr = SegDataset(TRAIN_IMAGES_DIRM, TRAIN_MASKS_DIRM, transform=train_tf)
    ds_mag_val = SegDataset(VAL_IMAGES_DIRM, VAL_MASKS_DIRM, transform=test_tf)

    ld_tr = DataLoader(ds_mag_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    ld_va = DataLoader(ds_mag_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FT, weight_decay=1e-4)
    train_loop(model, ld_tr, ld_va, opt, FT_EPOCHS, dev, PESOS_FT_BEST)
    torch.save(model.state_dict(), PESOS_FT_LAST)

    thr_ft = find_best_threshold(model, ld_va, dev)
    st_ft = evaluate(model, ld_test_mag, dev, thr=thr_ft)
    save_csv(st_ft, DIR_FT, "metricas_ft.csv")
    save_examples(model, ds_test_mag, dev, DIR_FT, "ft_", 5, thr=thr_ft)

    # 2) Consolidation on mixed (unfreeze all)
    print("\nConsolidation (Nopal + Maguey)...")
    freeze_encoder_rasem(model, freeze=False)

    ds_n_tr = SegDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, transform=train_tf)
    ds_n_val = SegDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, transform=test_tf)
    ds_mix_tr = ConcatDataset([ds_n_tr, ds_mag_tr])
    ds_mix_va = ConcatDataset([ds_n_val, ds_mag_val])

    ld_mix_tr = DataLoader(ds_mix_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    ld_mix_va = DataLoader(ds_mix_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    opt2 = optim.AdamW(model.parameters(), lr=LR_CONS, weight_decay=1e-4)
    train_loop(model, ld_mix_tr, ld_mix_va, opt2, CONS_EPOCHS, dev, PESOS_CONS_BEST)
    torch.save(model.state_dict(), PESOS_CONS_LAST)

    # Final tests
    ds_test_nop = SegDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR, transform=test_tf)
    ld_test_nop = DataLoader(ds_test_nop, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    thr_cons = find_best_threshold(model, ld_mix_va, dev)

    st_fin_mag = evaluate(model, ld_test_mag, dev, thr=thr_cons)
    st_fin_nop = evaluate(model, ld_test_nop, dev, thr=thr_cons)

    save_csv(st_fin_mag, DIR_CONS, "metricas_maguey.csv")
    save_csv(st_fin_nop, DIR_CONS, "metricas_nopal.csv")
    save_examples(model, ds_test_mag, dev, DIR_CONS, "mag_fin_", 3, thr=thr_cons)
    save_examples(model, ds_test_nop, dev, DIR_CONS, "nop_fin_", 3, thr=thr_cons)

    print("\nProcess completed:", PESOS_FT_BEST, "| CONS(best):", PESOS_CONS_BEST)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        default=os.path.join("Modelos", "modelo_best.pth"),
        help="Initial RASEM weights pretrained on Nopal"
    )
    args = ap.parse_args()
    main(args.weights)