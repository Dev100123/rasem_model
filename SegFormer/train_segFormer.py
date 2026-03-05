"""
Train SegFormer
"""

import os, time, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tv
import cv2
import matplotlib.pyplot as plt
from config import (
    TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR,
    VAL_IMAGES_DIR,   VAL_MASKS_DIR,
    TEST_IMAGES_DIR,  TEST_MASKS_DIR,
    BATCH_SIZE, NUM_WORKERS, NUM_CLASSES, IMAGE_SIZE,
    LEARNING_RATE, NUM_EPOCHS, MEAN, STD, OUT_DIR
)
from transformers import SegformerForSemanticSegmentation

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
        self.imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                            if f.lower().endswith((".png",".jpg",".jpeg"))])
        self.msks = sorted([os.path.join(msk_dir, f) for f in os.listdir(msk_dir)
                            if f.lower().endswith((".png",".jpg",".jpeg"))])
        assert len(self.imgs) == len(self.msks) and len(self.imgs) > 0, "Dataset vacío o desparejo."
        self.augment = augment
        self.to_tensor = tv.ToTensor()
        self.norm = tv.Normalize(mean=MEAN, std=STD)

    def _geom_aug(self, img, msk):
        if random.random() > 0.5:
            img = np.fliplr(img).copy(); msk = np.fliplr(msk).copy()
        if random.random() > 0.5:
            img = np.flipud(img).copy(); msk = np.flipud(msk).copy()
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k).copy(); msk = np.rot90(msk, k).copy()
        return img, msk

    def _photo_aug(self, img, p=0.3, b_lim=0.2, c_lim=0.2):

        if random.random() >= p:
            return img
        imgf = img.astype(np.float32)
        mean = imgf.mean(axis=(0, 1), keepdims=True)     # promedio por canal
        c = 1.0 + random.uniform(-c_lim, c_lim)          # contraste
        b = 1.0 + random.uniform(-b_lim, b_lim)          # brillo
        imgf = (imgf - mean) * c + mean                  # contraste
        imgf = imgf * b                                  # brillo
        return np.clip(imgf, 0, 255).astype(np.uint8)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.msks[idx], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, msk = self._geom_aug(img, msk)
            img = self._photo_aug(img)   # ← NUEVO: jitter brillo/contraste (p=0.3)

        img_raw = img.copy()
        msk_raw = ((msk > 127).astype(np.uint8) * 255)

        img_t = self.to_tensor(img)      # [0,1]
        img_t = self.norm(img_t)         # normalización ImageNet (0–1)
        msk_t = torch.from_numpy((msk > 127).astype(np.uint8)).long()

        return {"image": img_t, "mask": msk_t, "image_raw": img_raw, "mask_raw": msk_raw}
# ---------------- model ----------------
def up_to_mask(p, t):
    return torch.nn.functional.interpolate(p, size=t.shape[-2:], mode='bilinear', align_corners=False)

@torch.no_grad()
def compute_metrics(logits, masks, thr=0.5):
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = up_to_mask(logits, masks)
    prob = torch.sigmoid(logits[:, 0])
    pred = (prob > thr).float()
    maskf = masks.float()
    acc  = (pred == maskf).float().mean()
    rmse = torch.sqrt(((prob - maskf) ** 2).mean())
    inter= (pred * maskf).sum()
    union= pred.sum() + maskf.sum() - inter + 1e-6
    iou  = inter / union
    return acc.item(), rmse.item(), iou.item()

def dice_loss(p, t, eps=1e-6):
    if p.shape[-2:] != t.shape[-2:]:
        p = up_to_mask(p, t)
    prob = torch.sigmoid(p)
    t    = t.float().unsqueeze(1)
    inter= (prob * t).sum((1,2,3))
    union= prob.sum((1,2,3)) + t.sum((1,2,3)) + eps
    return 1 - (2 * inter / union).mean()

def combined_loss(p, t, bce):
    if p.shape[-2:] != t.shape[-2:]:
        p = up_to_mask(p, t)
    return bce(p, t.float().unsqueeze(1)) + 0.5 * dice_loss(p, t)

def save_panel(img_rgb, mask_gt, logits, out_path, thr=0.5):
    if logits.ndim == 3:
        logits = logits.squeeze(0)

    # Reescalar logits al tamaño de la máscara GT
    H, W = mask_gt.shape
    logits = torch.nn.functional.interpolate(
        logits.unsqueeze(0).unsqueeze(0),  # [1,1,h,w]
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob > thr).astype(np.uint8) * 255

    overlay = img_rgb.copy()
    overlay[pred == 255] = (255, 0, 0)

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(img_rgb); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(mask_gt, cmap="gray"); ax[1].set_title("GT"); ax[1].axis("off")
    ax[2].imshow(pred, cmap="gray"); ax[2].set_title("Pred"); ax[2].axis("off")
    ax[3].imshow(overlay); ax[3].set_title("Overlay"); ax[3].axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

# ---------------- train + Test ----------------
def main():
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    os.makedirs(OUT_DIR, exist_ok=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    bce = nn.BCEWithLogitsLoss()

    tr_ds = NopalDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, augment=True)
    va_ds = NopalDataset(VAL_IMAGES_DIR,   VAL_MASKS_DIR,   augment=False)
    te_ds = NopalDataset(TEST_IMAGES_DIR,  TEST_MASKS_DIR,  augment=False)

    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    hist = {k: [] for k in [
        "train_loss","train_acc","train_rmse","train_iou",
        "val_loss","val_acc","val_rmse","val_iou","t_img"
    ]}
    best_iou = -1.0

    for ep in range(1, NUM_EPOCHS + 1):
        # ---- Entrenamiento ----
        model.train()
        tl = ta = tr = ti = 0.0
        for batch in tr_ld:
            img, msk = batch["image"].to(device), batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(pixel_values=img).logits
            loss   = combined_loss(logits, msk, bce)
            loss.backward()
            opt.step()
            l = loss.item()
            a, r, i = compute_metrics(logits, msk)
            tl += l; ta += a; tr += r; ti += i

        ntr = len(tr_ld)
        tl, ta, tr, ti = [x / ntr for x in (tl, ta, tr, ti)]

        # ---- Validación ----
        model.eval()
        vl = va = vr = vi = 0.0
        t_sum = 0.0; n_img = 0
        with torch.no_grad():
            for batch in va_ld:
                img, msk = batch["image"].to(device), batch["mask"].to(device)
                t0 = time.perf_counter()
                logits = model(pixel_values=img).logits
                t_sum += (time.perf_counter() - t0); n_img += img.size(0)
                l = combined_loss(logits, msk, bce).item()
                a, r, i = compute_metrics(logits, msk)
                vl += l; va += a; vr += r; vi += i

        nva = len(va_ld)
        vl, va, vr, vi = [x / nva for x in (vl, va, vr, vi)]
        t_img = t_sum / n_img

        if vi > best_iou:
            best_iou = vi
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "modelo_best.pth"))

        for k, v in zip(hist.keys(), [tl, ta, tr, ti, vl, va, vr, vi, t_img]):
            hist[k].append(v)

        print(f"Ep {ep:03d}/{NUM_EPOCHS} | Train L:{tl:.4f} Acc:{ta:.4f} IoU:{ti:.4f} || Val L:{vl:.4f} Acc:{va:.4f} IoU:{vi:.4f}")

    #torch.save(model.state_dict(), os.path.join(OUT_DIR, "modelo_final.pth"))

    with open(os.path.join(OUT_DIR, "metricas_resumen.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metrica", "media", "desv_std"])
        for k in hist.keys():
            w.writerow([k, float(np.mean(hist[k])), float(np.std(hist[k], ddof=1))])

    # ---- TEST ----
    best_ckpt = os.path.join(OUT_DIR, "modelo_best.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    test_out_dir = os.path.join(OUT_DIR, "Test_SegFormer")
    os.makedirs(test_out_dir, exist_ok=True)

    logs = {k: [] for k in ("loss","acc","rmse","iou","t_img")}
    with torch.no_grad():
        for batch in te_ld:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            t0 = time.perf_counter()
            logits = model(pixel_values=img).logits
            dt = time.perf_counter() - t0
            l  = combined_loss(logits, msk, bce).item()
            a, r, i = compute_metrics(logits, msk)
            logs["loss"].append(l)
            logs["acc"].append(a)
            logs["rmse"].append(r)
            logs["iou"].append(i)
            logs["t_img"].append(dt / img.size(0))

    with open(os.path.join(test_out_dir, "test_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica","media","desv_std"])
        for k in logs.keys():
            w.writerow([k, float(np.mean(logs[k])), float(np.std(logs[k], ddof=1))])

    idxs = random.sample(range(len(te_ds)), k=min(3, len(te_ds)))
    for j, idx in enumerate(idxs):
        sample = te_ds[idx]
        x = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(pixel_values=x).logits.cpu().squeeze(0)[0]
        save_panel(sample["image_raw"], sample["mask_raw"], logit,
                   os.path.join(test_out_dir, f"vis_{j}.png"), thr=0.5)

if __name__ == "__main__":
    main()
