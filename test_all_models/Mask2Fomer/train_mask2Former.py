import os, time, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tv
import cv2
import matplotlib.pyplot as plt

from config_mask2former import (
    TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR,
    VAL_IMAGES_DIR,   VAL_MASKS_DIR,
    TEST_IMAGES_DIR,  TEST_MASKS_DIR,
    BATCH_SIZE, NUM_WORKERS, NUM_CLASSES, IMAGE_SIZE,
    LEARNING_RATE, NUM_EPOCHS, MEAN, STD, OUT_DIR
)

from transformers import Mask2FormerForUniversalSegmentation

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

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.msks[idx], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, msk = self._geom_aug(img, msk)

        img_raw = img.copy()
        msk_raw = ((msk > 127).astype(np.uint8) * 255)

        img_t = self.to_tensor(img)
        img_t = self.norm(img_t)
        msk_t = torch.from_numpy((msk > 127).astype(np.uint8)).long()

        return {"image": img_t, "mask": msk_t, "image_raw": img_raw, "mask_raw": msk_raw}

# --------------------------------
def up_to_mask(p, t_hw):

    if p.ndim == 3:  # [B,h,w] -> [B,1,h,w]
        p = p.unsqueeze(1)
    return torch.nn.functional.interpolate(p, size=t_hw, mode='bilinear', align_corners=False)

def compute_binary_logit_from_outputs(outputs):

    class_logits = getattr(outputs, "pred_logits", None)
    masks_logits = getattr(outputs, "pred_masks", None)

    if class_logits is None:
        class_logits = getattr(outputs, "class_queries_logits", None)
    if masks_logits is None:
        masks_logits = getattr(outputs, "masks_queries_logits", None)

    if class_logits is None or masks_logits is None:

        if masks_logits is None:
            raise RuntimeError("No se encontraron tensores de máscaras en las salidas del modelo.")
        # [B,Q,h,w] -> max sobre Q
        bin_logit = masks_logits.max(dim=1).values  # [B,h,w]
        return bin_logit

    B, Q, *_ = masks_logits.shape

    C = class_logits.shape[-1]

    cls_prob = torch.softmax(class_logits, dim=-1)  # [B,Q,C]

    if C == 1:
        weights = torch.ones((B, Q, 1), device=masks_logits.device, dtype=masks_logits.dtype)
    else:

        if C == 2:
            fg_idx = 1
            weights = cls_prob[:, :, fg_idx:fg_idx+1]  # [B,Q,1]
        else:
            weights = cls_prob[:, :, :-1].max(dim=-1, keepdim=True).values  # [B,Q,1]

    masks = masks_logits  # [B,Q,h,w]
    weights = weights.transpose(1, 2)  # [B,1,Q]

    weights = weights.unsqueeze(-1).unsqueeze(-1)     # [B,1,Q,1,1]
    masks   = masks.unsqueeze(1)                      # [B,1,Q,h,w]
    combined = (weights * masks).sum(dim=2)           # [B,1,h,w]
    bin_logit = combined.squeeze(1)                   # [B,h,w]
    return bin_logit

@torch.no_grad()
def compute_metrics_from_logit(bin_logit, masks, thr=0.5):

    B, H, W = masks.shape
    if bin_logit.shape[-2:] != (H, W):
        bin_logit = up_to_mask(bin_logit, (H, W)).squeeze(1)  # -> [B,h,w]

    prob = torch.sigmoid(bin_logit)
    pred = (prob > thr).float()
    maskf = masks.float()

    acc  = (pred == maskf).float().mean()
    rmse = torch.sqrt(((prob - maskf) ** 2).mean())

    inter = (pred * maskf).sum()
    union = pred.sum() + maskf.sum() - inter + 1e-6
    iou   = inter / union
    return acc.item(), rmse.item(), iou.item()

def dice_loss_from_logit(bin_logit, masks, eps=1e-6):

    B, H, W = masks.shape
    if bin_logit.shape[-2:] != (H, W):
        bin_logit = up_to_mask(bin_logit, (H, W)).squeeze(1)

    prob = torch.sigmoid(bin_logit)
    t    = masks.float()
    inter= (prob * t).sum((1,2))
    union= prob.sum((1,2)) + t.sum((1,2)) + eps
    return 1 - (2 * inter / union).mean()

def combined_loss_from_outputs(outputs, masks, bce):

    bin_logit = compute_binary_logit_from_outputs(outputs)  # [B,h,w]
    B, H, W = masks.shape
    if bin_logit.shape[-2:] != (H, W):
        bin_logit = up_to_mask(bin_logit, (H, W)).squeeze(1)  # [B,h,w]
    loss_bce = bce(bin_logit, masks.float())
    loss_dice = dice_loss_from_logit(bin_logit, masks)
    return loss_bce + 0.5 * loss_dice, bin_logit

def save_panel(img_rgb, mask_gt, bin_logit, out_path, thr=0.5):

    H, W = mask_gt.shape
    if bin_logit.shape[-2:] != (H, W):
        bin_logit = torch.nn.functional.interpolate(bin_logit.unsqueeze(0).unsqueeze(0),
                                                    size=(H, W), mode='bilinear',
                                                    align_corners=False).squeeze(0).squeeze(0)
    prob = torch.sigmoid(bin_logit).cpu().numpy()
    pred = (prob > thr).astype(np.uint8) * 255

    overlay = img_rgb.copy()
    overlay[pred == 255] = (255, 0, 0)

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(img_rgb); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(mask_gt, cmap="gray"); ax[1].set_title("GT"); ax[1].axis("off")
    ax[2].imshow(pred, cmap="gray"); ax[2].set_title("Pred"); ax[2].axis("off")
    ax[3].imshow(overlay); ax[3].set_title("Overlay"); ax[3].axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

# ---------------- Train + Test ----------------
def main():
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    os.makedirs(OUT_DIR, exist_ok=True)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-ade-semantic",
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

    # --------- Entrenamiento ---------
    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        tl = ta = tr = ti = 0.0

        for batch in tr_ld:
            img, msk = batch["image"].to(device), batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            outputs = model(pixel_values=img)
            loss, bin_logit = combined_loss_from_outputs(outputs, msk, bce)
            loss.backward()
            opt.step()

            l = loss.item()
            a, r, i = compute_metrics_from_logit(bin_logit.detach(), msk)
            tl += l; ta += a; tr += r; ti += i

        ntr = len(tr_ld)
        tl, ta, tr, ti = [x / ntr for x in (tl, ta, tr, ti)]

        # --------- Validación ---------
        model.eval()
        vl = va = vr = vi = 0.0
        t_sum = 0.0; n_img = 0
        with torch.no_grad():
            for batch in va_ld:
                img, msk = batch["image"].to(device), batch["mask"].to(device)
                t0 = time.perf_counter()
                outputs = model(pixel_values=img)
                t_sum += (time.perf_counter() - t0); n_img += img.size(0)
                loss_val, bin_logit = combined_loss_from_outputs(outputs, msk, bce)
                a, r, i = compute_metrics_from_logit(bin_logit, msk)
                vl += loss_val.item(); va += a; vr += r; vi += i

        nva = len(va_ld)
        vl, va, vr, vi = [x / nva for x in (vl, va, vr, vi)]
        t_img = t_sum / max(1, n_img)

        if vi > best_iou:
            best_iou = vi
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "modelo_best.pth"))

        for k, v in zip(hist.keys(), [tl, ta, tr, ti, vl, va, vr, vi, t_img]):
            hist[k].append(v)

        print(f"Ep {ep:03d}/{NUM_EPOCHS} | Train L:{tl:.4f} Acc:{ta:.4f} IoU:{ti:.4f} || Val L:{vl:.4f} Acc:{va:.4f} IoU:{vi:.4f}")

    # --------- Resumen de métricas (media ± sd) ---------
    with open(os.path.join(OUT_DIR, "metricas_resumen.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metrica", "media", "desv_std"])
        for k in hist.keys():
            w.writerow([k, float(np.mean(hist[k])), float(np.std(hist[k], ddof=1))])

    # --------- TEST ---------
    best_ckpt = os.path.join(OUT_DIR, "modelo_best.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    test_out_dir = os.path.join(OUT_DIR, "Test_Mask2Former")
    os.makedirs(test_out_dir, exist_ok=True)

    logs = {k: [] for k in ("loss","acc","rmse","iou","t_img")}
    with torch.no_grad():
        for batch in te_ld:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            t0 = time.perf_counter()
            outputs = model(pixel_values=img)
            dt = time.perf_counter() - t0
            loss_test, bin_logit = combined_loss_from_outputs(outputs, msk, bce)
            a, r, i = compute_metrics_from_logit(bin_logit, msk)
            logs["loss"].append(loss_test.item())
            logs["acc"].append(a)
            logs["rmse"].append(r)
            logs["iou"].append(i)
            logs["t_img"].append(dt / img.size(0))

    with open(os.path.join(test_out_dir, "test_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica","media","desv_std"])
        for k in logs.keys():
            w.writerow([k, float(np.mean(logs[k])), float(np.std(logs[k], ddof=1))])

    # --------- Visual ---------
    idxs = random.sample(range(len(te_ds)), k=min(3, len(te_ds)))
    for j, idx in enumerate(idxs):
        sample = te_ds[idx]
        x = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=x)
            bin_logit = compute_binary_logit_from_outputs(outputs).squeeze(0)  # [h,w]
        save_panel(sample["image_raw"], sample["mask_raw"], bin_logit,
                   os.path.join(test_out_dir, f"vis_{j}.png"), thr=0.5)

if __name__ == "__main__":
    main()