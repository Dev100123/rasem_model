"""
FINE TUNNING 

Swin-UNet Expansion — Nopal → Maguey → Nopal+Maguey
Phases
0) Pre-test: Direct inference on Maguey dataset.
1) Fine-Tuning: Freeze backbone layers 0-3 and train for 30 epochs on Maguey dataset.
2) Consolidation: Unfreeze all layers and train for the Mixed (Nopal+Maguey) dataset.
"""

# ───────── Imports ─────────
import os, time, random, math, argparse, csv
import numpy as np, cv2
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T

# ───────── Config (rutas) ─────────
from config import (
    TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR,    # nopal train
    VAL_IMAGES_DIR,   VAL_MASKS_DIR,      # nopal val
    TEST_IMAGES_DIR,  TEST_MASKS_DIR,     # nopal test
    TRAIN_IMAGES_DIRM, TRAIN_MASKS_DIRM,  # maguey train
    VAL_IMAGES_DIRM,   VAL_MASKS_DIRM,    # maguey val
    TEST_IMAGES_DIRM,  TEST_MASKS_DIRM,   # maguey test
    BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE, LEARNING_RATE
)
from dataset import NopalDataset
from model   import SwinUNet

# ───────── Ajustes globales ─────────

FT_EPOCHS, CONS_EPOCHS = 30, 10
LR_FT,   LR_CONS       = LEARNING_RATE, 1e-5
PESOS_FT, PESOS_CONS   = f"pesos_ft.pth", f"modelo_final.pth"

DIR_PRETEST = "Test_Maguey"
DIR_FT      = "Test_Fine_Tuning"
DIR_CONS    = "Test_Nopal+Maguey"
for d in (DIR_PRETEST, DIR_FT, DIR_CONS):
    os.makedirs(d, exist_ok=True)

# ───────── Transformaciones ─────────
class RandomFlipRotate:
    def __call__(self, s):
        img, msk = s["image"], s["mask"]
        if random.random() > .5: img, msk = np.fliplr(img), np.fliplr(msk)
        if random.random() > .5: img, msk = np.flipud(img), np.flipud(msk)
        k = random.choice([0, 1, 2])
        if k: img, msk = np.rot90(img, k), np.rot90(msk, k)
        return {"image": img.copy(), "mask": msk.copy()}

class ToTensorResize:
    def __call__(self, s):
        img = cv2.resize(s["image"], IMAGE_SIZE)
        msk = cv2.resize(s["mask"],  IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        img = (img.astype(np.float32) / 255.).transpose(2, 0, 1)
        return {"image": torch.tensor(img, dtype=torch.float32),
                "mask":  torch.tensor(msk, dtype=torch.long)}

train_tf = T.Compose([RandomFlipRotate(), ToTensorResize()])
test_tf  = T.Compose([ToTensorResize()])

# ───────── Métricas y helpers ─────────
def _iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or (pred, gt).sum() + 1e-6
    return inter / union

def _rmse(pred, gt):
    return math.sqrt(((pred.astype(float) - gt)**2).mean())

def evaluate(model, loader, device):
    model.eval()
    IoU, RM, AC, Ti = [], [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device)
            gt = (b["mask"].numpy() > 0)
            t0 = time.perf_counter()
            out = model(x)
            torch.cuda.synchronize() if device.type == "cuda" else None
            Ti.append(time.perf_counter() - t0)
            pr = (torch.sigmoid(out)[:, 0] > 0.5).cpu().numpy()
            IoU.append(_iou(pr, gt))
            RM.append(_rmse(pr, gt))
            AC.append(np.mean(pr == gt))
    return {
        "IoU_mean": np.mean(IoU),  "IoU_std": np.std(IoU,  ddof=1),
        "RMSE_mean": np.mean(RM),  "RMSE_std": np.std(RM,  ddof=1),
        "Acc_mean": np.mean(AC),   "Acc_std":  np.std(AC,  ddof=1),
        "t_img_mean": np.mean(Ti), "t_img_std":np.std(Ti, ddof=1)
    }

def save_csv(stats, out_dir, name):
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "media", "desv_std"])
        for m in ["IoU", "RMSE", "Acc", "t_img"]:
            w.writerow([m, stats[f"{m}_mean"], stats[f"{m}_std"]])

def save_examples(model, ds, device, out_dir, prefix, n=5):
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= n: break
            x  = b["image"].to(device)
            gt = (b["mask"].numpy() > 0)[0]
            pr = (torch.sigmoid(model(x))[:,0] > 0.5).cpu().numpy()[0]

            img = (x.squeeze().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
            gt_v = (gt*255).astype(np.uint8)
            pr_v = (pr*255).astype(np.uint8)
            ov   = img.copy()
            ov[pr] = (0.5*np.array([255,0,0]) + 0.5*ov[pr]).astype(np.uint8)

            panel = np.hstack([img,
                               cv2.cvtColor(gt_v, cv2.COLOR_GRAY2BGR),
                               cv2.cvtColor(pr_v, cv2.COLOR_GRAY2BGR),
                               ov])
            cv2.imwrite(os.path.join(out_dir, f"{prefix}{i}.png"),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

def freeze_backbone(model, freeze=True):
    for n, p in model.backbone.named_parameters():
        if any(n.startswith(f"stages.{i}") for i in range(3)):
            p.requires_grad = not freeze

def combined_loss(logit, tgt):
    ce = nn.BCEWithLogitsLoss()(logit[:,0], tgt.float())
    pr = torch.sigmoid(logit[:,0])
    inter = (pr*tgt).sum((1,2)); union = pr.sum((1,2)) + tgt.sum((1,2)) + 1e-6
    dice = 1 - (2*inter/union).mean()
    return ce + 0.5*dice

def train_loop(model, loader, opt, epochs, device):
    for e in range(1, epochs+1):
        model.train(); running = 0
        for b in loader:
            x=b["image"].to(device); y=b["mask"].to(device)
            opt.zero_grad()
            loss = combined_loss(model(x), y.float())
            loss.backward(); opt.step(); running += loss.item()
        print(f"Época {e:02d}/{epochs}  Loss {running/len(loader):.4f}")

# ───────── Main ─────────
def main(init_weights):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNet(num_classes=1).to(dev)
    model.load_state_dict(torch.load(init_weights, map_location=dev))

    # 0) Inferencia inicial (maguey)
    ds_test_mag = NopalDataset(TEST_IMAGES_DIRM, TEST_MASKS_DIRM, test_tf)
    st_pre = evaluate(model,
        DataLoader(ds_test_mag, batch_size=1, shuffle=False,
                   num_workers=NUM_WORKERS), dev)
    save_csv(st_pre, DIR_PRETEST, f"metricas_pre.csv")
    save_examples(model, ds_test_mag, dev, DIR_PRETEST, "mag_", 5)

    # 1) Fine-Tuning
    print("\n⏩ Fine-Tuning …")
    freeze_backbone(model, True)
    ds_mag_tr  = NopalDataset(TRAIN_IMAGES_DIRM, TRAIN_MASKS_DIRM, train_tf)
    ds_mag_val = NopalDataset(VAL_IMAGES_DIRM,   VAL_MASKS_DIRM,   test_tf)

    ld_tr = DataLoader(ds_mag_tr,  batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS)
    ld_va = DataLoader(ds_mag_val, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS)

    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=LR_FT)
    train_loop(model, ld_tr, opt, FT_EPOCHS, dev)
    torch.save(model.state_dict(), PESOS_FT)

    st_ft = evaluate(model,
        DataLoader(ds_test_mag, batch_size=1, shuffle=False,
                   num_workers=NUM_WORKERS), dev)
    save_csv(st_ft, DIR_FT, f"metricas_ft.csv")
    save_examples(model, ds_test_mag, dev, DIR_FT, "ft_", 5)

    # 2) Consolidación
    print("\n⏩ Consolidación …")
    freeze_backbone(model, False)

    ds_nopal_tr = NopalDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, train_tf)
    ds_mix_tr   = ConcatDataset([ds_nopal_tr, ds_mag_tr])
    ld_mix_tr   = DataLoader(ds_mix_tr, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS)

    ds_nopal_val = NopalDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, test_tf)
    ds_mag_val   = NopalDataset(VAL_IMAGES_DIRM, VAL_MASKS_DIRM, test_tf)
    ds_mix_val   = ConcatDataset([ds_nopal_val, ds_mag_val])
    ld_mix_val   = DataLoader(ds_mix_val, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    opt2 = optim.Adam(model.parameters(), lr=LR_CONS)
    train_loop(model, ld_mix_tr, opt2, CONS_EPOCHS, dev)
    torch.save(model.state_dict(), PESOS_CONS)

    # Métricas finales
    ds_test_nopal = NopalDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR, test_tf)
    st_fin_mag = evaluate(model,
        DataLoader(ds_test_mag, batch_size=1, shuffle=False,
                   num_workers=NUM_WORKERS), dev)
    st_fin_nop = evaluate(model,
        DataLoader(ds_test_nopal, batch_size=1, shuffle=False,
                   num_workers=NUM_WORKERS), dev)

    save_csv(st_fin_mag, DIR_CONS, f"metricas_maguey.csv")
    save_csv(st_fin_nop, DIR_CONS, f"metricas_nopal.csv")
    save_examples(model, ds_test_mag,   dev, DIR_CONS, "mag_fin_", 3)
    save_examples(model, ds_test_nopal, dev, DIR_CONS, "nop_fin_", 3)

    print("\nProceso completo ✅  Pesos consolidados →", PESOS_CONS)

# ───────── Ejecutar ─────────
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default=os.path.join("Modelos", f"modelo_best.pth"),
                    help="Pesos iniciales entrenados en nopal")
    main(ap.parse_args().weights)