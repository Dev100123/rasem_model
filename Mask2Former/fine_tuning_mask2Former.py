"""
FINE TUNNING 

Mask2Former Expansion — Nopal → Maguey → Nopal+Maguey
Phases
0) Pre-test: Direct inference on Maguey dataset.
1) Fine-Tuning: Freeze backbone layers on Maguey dataset.
2) Consolidation: Unfreeze all layers and train Mixed (Nopal+Maguey) dataset.
"""


# ───────── Imports ─────────
import os, time, random, math, argparse, csv
import numpy as np, cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from contextlib import nullcontext

# HF Transformers (Mask2Former universal)
from transformers import Mask2FormerForUniversalSegmentation

# ───────── Config (rutas y params) ─────────
from config import (
    # Nopal
    TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR,
    VAL_IMAGES_DIR,   VAL_MASKS_DIR,
    TEST_IMAGES_DIR,  TEST_MASKS_DIR,
    # Maguey
    TRAIN_IMAGES_DIRM, TRAIN_MASKS_DIRM,
    VAL_IMAGES_DIRM,   VAL_MASKS_DIRM,
    TEST_IMAGES_DIRM,  TEST_MASKS_DIRM,
    # Hyperparameters
    BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE, LEARNING_RATE
)

#dataset for Mask2Former
from train_mask2Former import (
    NopalDataset,
    compute_binary_logit_from_outputs,
    dice_loss_from_logit,
    up_to_mask
)

# ───────── OUT ─────────
DIR_PRETEST = "Test_Maguey"         # Fase 0
DIR_FT      = "Test_Fine_Tuning"    # Fase 1
DIR_CONS    = "Test_Nopal+Maguey"   # Fase 2
for d in (DIR_PRETEST, DIR_FT, DIR_CONS):
    os.makedirs(d, exist_ok=True)

# ───────── Epochs y LR ─────────
FT_EPOCHS,   CONS_EPOCHS = 30, 10
LR_FT,       LR_CONS     = LEARNING_RATE, 1e-5
PESOS_FT_BEST,  PESOS_CONS_BEST  = "mask2former_ft_best.pth",  "mask2former_cons_best.pth"
PESOS_FT_LAST,  PESOS_CONS_LAST  = "mask2former_ft_last.pth",  "mask2former_cons_last.pth"

# ───────── Seed ─────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(42)

# ───────── Metrics and helpers ─────────
def _iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or (pred, gt).sum() + 1e-6
    return inter / union

def _rmse(pred, gt):
    return math.sqrt(((pred.astype(float) - gt)**2).mean())

@torch.no_grad()
def evaluate_mask2former(model, loader, device, thr=0.5, use_amp=True):
    model.eval()
    IoU, RM, AC, Ti = [], [], [], []
    amp_enabled = (use_amp and device.type == "cuda")
    autocast_ctx = (lambda: torch.amp.autocast('cuda')) if amp_enabled else nullcontext
    for b in loader:
        x  = b["image"].to(device, non_blocking=True)          # (B,3,H,W)
        gt = (b["mask"].cpu().numpy() > 0)                     # (B,H,W) en CPU

        t0 = time.perf_counter()
        with autocast_ctx():
            out = model(pixel_values=x)
            bin_logit = compute_binary_logit_from_outputs(out) # [B,h,w]
        
        if bin_logit.shape[-2:] != b["mask"].shape[-2:]:
            bin_logit = up_to_mask(bin_logit, b["mask"].shape[-2:]).squeeze(1)  # [B,h,w]
        if device.type == "cuda": torch.cuda.synchronize()
        Ti.append(time.perf_counter() - t0)

        prob = torch.sigmoid(bin_logit).cpu().numpy()
        pr   = (prob > thr).astype(np.uint8)                   # (B,H,W)
        IoU.append(_iou(pr, gt))
        RM.append(_rmse(pr, gt))
        AC.append(np.mean(pr == gt))

    return {
        "IoU_mean": np.mean(IoU),  "IoU_std":  np.std(IoU,  ddof=1),
        "RMSE_mean": np.mean(RM),  "RMSE_std": np.std(RM,  ddof=1),
        "Acc_mean": np.mean(AC),   "Acc_std":  np.std(AC,  ddof=1),
        "t_img_mean": np.mean(Ti), "t_img_std": np.std(Ti, ddof=1)
    }

def save_csv(stats, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "media", "desv_std"])
        for m in ["IoU", "RMSE", "Acc", "t_img"]:
            w.writerow([m, float(stats[f"{m}_mean"]), float(stats[f"{m}_std"])])

@torch.no_grad()
def save_examples_mask2former(model, ds, device, out_dir, prefix, n=5, thr=0.5, use_amp=True):
    os.makedirs(out_dir, exist_ok=True)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    amp_enabled = (use_amp and device.type == "cuda")
    autocast_ctx = (lambda: torch.amp.autocast('cuda')) if amp_enabled else nullcontext
    for i, b in enumerate(ld):
        if i >= n: break
        x = b["image"].to(device, non_blocking=True)                 # (1,3,H,W)
        img_raw = b["image_raw"][0].numpy() if isinstance(b["image_raw"], torch.Tensor) else b["image_raw"][0]
        gt_raw  = b["mask_raw"][0].numpy()  if isinstance(b["mask_raw"],  torch.Tensor) else b["mask_raw"][0]
        gt_bin  = (gt_raw > 0).astype(np.uint8)*255

        with autocast_ctx():
            out = model(pixel_values=x)
            bin_logit = compute_binary_logit_from_outputs(out)[0]    # [h,w]
        if bin_logit.shape[-2:] != gt_raw.shape[-2:]:
            bin_logit = torch.nn.functional.interpolate(
                bin_logit.unsqueeze(0).unsqueeze(0), size=gt_raw.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
        prob = torch.sigmoid(bin_logit).cpu().numpy()
        pr   = (prob > thr).astype(np.uint8)*255

        overlay = img_raw.copy()
        overlay[pr == 255] = (255, 0, 0)

        panel = np.hstack([
            img_raw,
            cv2.cvtColor(gt_bin, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(pr, cv2.COLOR_GRAY2BGR),
            overlay
        ])
        cv2.imwrite(os.path.join(out_dir, f"{prefix}{i}.png"),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

def combined_loss_from_outputs(outputs, masks, bce):
    
    bin_logit = compute_binary_logit_from_outputs(outputs)  # [B,h,w]
    B, H, W = masks.shape
    if bin_logit.shape[-2:] != (H, W):
        bin_logit = up_to_mask(bin_logit, (H, W)).squeeze(1)       # [B,h,w]
    loss_bce = bce(bin_logit, masks.float())
    loss_dice = dice_loss_from_logit(bin_logit, masks)
    return loss_bce + 0.5 * loss_dice, bin_logit

def bce_dice_train_step(outputs, target, bce):
    return combined_loss_from_outputs(outputs, target, bce)[0]

def freeze_mask2former_backbone(model, freeze=True):
    for n, p in model.named_parameters():
        name = n.lower()
        if ("backbone" in name) or ("pixel_decoder" in name) or ("pixeldecoder" in name):
            p.requires_grad = (not freeze)
        else:
            p.requires_grad = True  # decoder y heads

def find_best_threshold_mask2former(model, loader, device, use_amp=True):

    model.eval()
    thrs = np.linspace(0.30, 0.70, 9)
    scores = np.zeros_like(thrs, dtype=np.float64)
    amp_enabled = (use_amp and device.type == "cuda")
    autocast_ctx = (lambda: torch.amp.autocast('cuda')) if amp_enabled else nullcontext
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device, non_blocking=True)
            y = (b["mask"].cpu().numpy() > 0).astype(np.uint8)
            with autocast_ctx():
                out = model(pixel_values=x)
                bin_logit = compute_binary_logit_from_outputs(out)   # [B,h,w]
            if bin_logit.shape[-2:] != b["mask"].shape[-2:]:
                bin_logit = up_to_mask(bin_logit, b["mask"].shape[-2:]).squeeze(1)  # [B,h,w]
            prob = torch.sigmoid(bin_logit).cpu().numpy()
            for i, t in enumerate(thrs):
                pr = (prob > t).astype(np.uint8)
                inter = (pr & y).sum()
                uni   = pr.sum() + y.sum() - inter + 1e-6
                scores[i] += inter / uni
    return float(thrs[np.argmax(scores)])

def eval_one_epoch_mask2former(model, loader, device, thr=0.5):
    st = evaluate_mask2former(model, loader, device, thr=thr)
    return st["IoU_mean"]

def train_loop_mask2former(model, loader_tr, loader_va, opt, epochs, device, ckpt_path, use_amp=True):
    
    best_iou = -1.0
    amp_enabled = (use_amp and device.type == "cuda")
    autocast_ctx = (lambda: torch.amp.autocast('cuda')) if amp_enabled else nullcontext
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    bce = nn.BCEWithLogitsLoss()

    for e in range(1, epochs+1):
        model.train(); running = 0.0
        for b in loader_tr:
            x = b["image"].to(device, non_blocking=True)          # (B,3,H,W)
            y = b["mask"].to(device, non_blocking=True)           # (B,H,W)
            if y.dtype not in (torch.float32, torch.float64): y = y.float()
            if y.max() > 1.0: y = (y > 0).float()

            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                out = model(pixel_values=x)
                loss = bce_dice_train_step(out, y, bce)

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item()

        iou_val = eval_one_epoch_mask2former(model, loader_va, device, thr=0.5)
        print(f"Época {e:02d}/{epochs}  Loss {running/len(loader_tr):.4f}  IoU_val {iou_val:.4f}")
        if iou_val > best_iou:
            best_iou = iou_val
            torch.save(model.state_dict(), ckpt_path)

# ───────── Main ─────────
def main(init_weights):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instancia (misma arquitectura que en tu entrenamiento base)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-ade-semantic",
        ignore_mismatched_sizes=True
    ).to(dev)

    # Cargar pesos de nopal entrenados por ti con Mask2Former (strict=False por si cambia algún head)
    state = torch.load(init_weights, map_location=dev)
    model.load_state_dict(state, strict=False)

    # ── Datasets y loaders
    ds_test_mag = NopalDataset(TEST_IMAGES_DIRM, TEST_MASKS_DIRM, augment=False)
    ld_test_mag = DataLoader(ds_test_mag, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 0) Pre-test en maguey
    st_pre = evaluate_mask2former(model, ld_test_mag, dev, thr=0.5)
    save_csv(st_pre, DIR_PRETEST, "metricas_pre.csv")
    save_examples_mask2former(model, ds_test_mag, dev, DIR_PRETEST, "mag_", 5, thr=0.5)

    # 1) Fine-Tuning en maguey (congelando backbone + pixel_decoder)
    print("\n⏩ Fine-Tuning (maguey)…")
    freeze_mask2former_backbone(model, freeze=True)

    ds_mag_tr  = NopalDataset(TRAIN_IMAGES_DIRM, TRAIN_MASKS_DIRM, augment=True)
    ds_mag_val = NopalDataset(VAL_IMAGES_DIRM,   VAL_MASKS_DIRM,   augment=False)

    ld_tr = DataLoader(ds_mag_tr,  batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True)
    ld_va = DataLoader(ds_mag_val, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LR_FT, weight_decay=1e-4)
    train_loop_mask2former(model, ld_tr, ld_va, opt, FT_EPOCHS, dev, PESOS_FT_BEST)

    # Últimos pesos tras FT (además del best)
    torch.save(model.state_dict(), PESOS_FT_LAST)

    # Ajusta umbral con validación de maguey
    thr_ft = find_best_threshold_mask2former(model, ld_va, dev)
    st_ft  = evaluate_mask2former(model, ld_test_mag, dev, thr=thr_ft)
    save_csv(st_ft, DIR_FT, "metricas_ft.csv")
    save_examples_mask2former(model, ds_test_mag, dev, DIR_FT, "ft_", 5, thr=thr_ft)

    # 2) Consolidación (nopal + maguey)
    print("\n⏩ Consolidación (nopal+maguey)…")
    freeze_mask2former_backbone(model, freeze=False)  # desbloquear todo

    ds_n_tr   = NopalDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, augment=True)
    ds_n_val  = NopalDataset(VAL_IMAGES_DIR,   VAL_MASKS_DIR,   augment=False)
    ds_mix_tr = ConcatDataset([ds_n_tr, ds_mag_tr])
    ds_mix_va = ConcatDataset([ds_n_val, ds_mag_val])

    ld_mix_tr = DataLoader(ds_mix_tr, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
    ld_mix_va = DataLoader(ds_mix_va, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    opt2 = optim.AdamW(model.parameters(), lr=LR_CONS, weight_decay=1e-4)
    train_loop_mask2former(model, ld_mix_tr, ld_mix_va, opt2, CONS_EPOCHS, dev, PESOS_CONS_BEST)

    torch.save(model.state_dict(), PESOS_CONS_LAST)

    # Métricas finales post-consolidación (ambos cultivos)
    ds_test_nop = NopalDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR, augment=False)
    ld_test_nop = DataLoader(ds_test_nop, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Re-estimar umbral con validación mixta para uso equitativo
    thr_cons = find_best_threshold_mask2former(model, ld_mix_va, dev)

    st_fin_mag = evaluate_mask2former(model, ld_test_mag, dev, thr=thr_cons)
    st_fin_nop = evaluate_mask2former(model, ld_test_nop, dev, thr=thr_cons)

    save_csv(st_fin_mag, DIR_CONS, "metricas_maguey.csv")
    save_csv(st_fin_nop, DIR_CONS, "metricas_nopal.csv")
    save_examples_mask2former(model, ds_test_mag, dev, DIR_CONS, "mag_fin_", 3, thr=thr_cons)
    save_examples_mask2former(model, ds_test_nop, dev, DIR_CONS, "nop_fin_", 3, thr=thr_cons)

    print("\nProceso completo ✅  Pesos FT(best):", PESOS_FT_BEST, "| CONS(best):", PESOS_CONS_BEST)

# ───────── Ejecutar ─────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
        default=os.path.join("Modelos_Mask2Former", "modelo_best.pth"),
        help="Pesos iniciales entrenados en nopal (Mask2Former universal)"
    )
    main(ap.parse_args().weights)