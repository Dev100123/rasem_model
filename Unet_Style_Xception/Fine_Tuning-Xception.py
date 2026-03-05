# -*- coding: utf-8 -*-
"""
UNet‑Style‑Xception
Expansion — Nopal → Maguey → Nopal+Maguey
Phases
0) Pre-test: Direct inference on Maguey dataset.
1) Fine-Tuning: Freeze backbone layers 0-3 and train for 30 epochs on Maguey dataset.
2) Consolidation: Unfreeze all layers and train for the Mixed (Nopal+Maguey) dataset.

"""
import os, random, time, cv2, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from train_Xception import UNetXception, metrics, save_plot

# ─────────────── Path ────────────────
FT_DIR = os.path.join(config.PROJECT_DIR, "Fine‑Tuning_Xception")
CS_DIR = os.path.join(config.PROJECT_DIR, "Consolidation_Xception")
os.makedirs(FT_DIR, exist_ok=True); os.makedirs(CS_DIR, exist_ok=True)

# ──────────────────── Aux ────────────────────
def save_txt(path, dct, header=None):
    with open(path, "w", encoding="utf-8") as f:
        if header: f.write(header + "\n")
        for k, v in dct.items():
            f.write(f"{k}:{v:.6f}\n")

def run_epoch(model, loader, criterion, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    agg = {k: 0 for k in ["loss", "acc", "rmse", "iou"]}
    t_tot, n_img = 0.0, 0
    for imgs, msks in (pbar := tqdm(loader, leave=False)):
        imgs, msks = imgs.to(config.DEVICE), msks.to(config.DEVICE)
        tic = time.perf_counter()
        with torch.set_grad_enabled(train):
            outs = model(imgs)
            loss = criterion(outs, msks)
            if train:
                optim.zero_grad(); loss.backward(); optim.step()
        torch.cuda.synchronize() if config.DEVICE == "cuda" else None
        t_tot += time.perf_counter() - tic; n_img += imgs.size(0)

        acc, rmse, iou = metrics(outs.detach(), msks)
        for k, v in zip(["loss", "acc", "rmse", "iou"],
                        [loss.item(), acc, rmse, iou]):
            agg[k] += v
        pbar.set_description(f"L:{loss.item():.3f} IoU:{iou:.3f}")

    n = len(loader)
    agg = {k: v / n for k, v in agg.items()}
    agg["t_img"] = t_tot / n_img
    return agg

# ────────────────── Dataset ──────────────────
class SimpleDataset(Dataset):
    def __init__(self, img_dir, msk_dir):
        self.ip = sorted([os.path.join(img_dir, f)
                          for f in os.listdir(img_dir) if f.endswith(".png")])
        self.mp = sorted([os.path.join(msk_dir, f)
                          for f in os.listdir(msk_dir) if f.endswith(".png")])

    def __len__(self): return len(self.ip)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.ip[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.mp[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, config.IMAGE_SIZE[::-1]) / 255.0
        msk = (cv2.resize(msk, config.IMAGE_SIZE[::-1]) / 255.0 > 0.5).astype(np.float32)
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        msk = torch.tensor(msk, dtype=torch.float32).unsqueeze(0)
        return img, msk

# ─────────────────────────── Main ────────────────────────────
def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    # ─── datasets (nopal y maguey) ───
    n_tr = SimpleDataset(config.TRAIN_IMAGES_DIR,  config.TRAIN_MASKS_DIR)
    n_vl = SimpleDataset(config.VAL_IMAGES_DIR,    config.VAL_MASKS_DIR)
    n_te = SimpleDataset(config.TEST_IMAGES_DIR,   config.TEST_MASKS_DIR)

    m_tr = SimpleDataset(config.TRAIN_IMAGES_DIRM, config.TRAIN_MASKS_DIRM)
    m_vl = SimpleDataset(config.VAL_IMAGES_DIRM,   config.VAL_MASKS_DIRM)
    m_te = SimpleDataset(config.TEST_IMAGES_DIRM,  config.TEST_MASKS_DIRM)

    dl = lambda ds, sh: DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=sh,
                                   num_workers=config.NUM_WORKERS, pin_memory=True)
    tr_dl_m, vl_dl_m, te_dl_m = dl(m_tr, True), dl(m_vl, False), dl(m_te, False)

    # ─── Model (nopal) ───
    model = UNetXception(config.NUM_CLASSES).to(config.DEVICE)
    base_ckpt = os.path.join(config.CHECKPOINT_DIR, "unetx_best.pth")
    model.load_state_dict(torch.load(base_ckpt, map_location=config.DEVICE))

    criterion = nn.BCEWithLogitsLoss()

    # ========== 1) baseline ==========
    direct = run_epoch(model, te_dl_m, criterion)
    save_txt(os.path.join(FT_DIR, "direct_metrics.txt"),
             direct, "# métricas inferencia directa (modelo de nopal)")
    for i in random.sample(range(len(m_te)), 3):
        img, msk = m_te[i]
        with torch.no_grad():
            logit = model(img.unsqueeze(0).to(config.DEVICE)).cpu().squeeze().numpy()
        save_plot((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                  (msk.squeeze().numpy() * 255).astype(np.uint8),
                  logit, os.path.join(FT_DIR, f"direct_vis_{i}.png"))

    # ========== 2) Fine‑tuning ==========
    for n, p in model.named_parameters():
        if n.split('.')[0] in {"e1", "e2", "e3"}:
            p.requires_grad = False
    optim_ft = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=1e-4, weight_decay=1e-5)

    EPOCHS_FT = 30
    best_ft_iou, hist_ft = 0, []
    for ep in range(1, EPOCHS_FT + 1):
        tr = run_epoch(model, tr_dl_m, criterion, optim_ft)
        vl = run_epoch(model, vl_dl_m, criterion)
        hist_ft.append({"epoch": ep, "train": tr, "val": vl})
        if vl["iou"] > best_ft_iou:
            best_ft_iou = vl["iou"]
            torch.save(model.state_dict(),
                       os.path.join(FT_DIR, "unetx_finetuned.pth"))
        print(f"FT {ep:2d}/{EPOCHS_FT} | val IoU:{vl['iou']:.4f}")

    # Test fine‑tuning
    model.load_state_dict(torch.load(os.path.join(FT_DIR, "unetx_finetuned.pth"),
                                     map_location=config.DEVICE))
    test_ft = run_epoch(model, te_dl_m, criterion)
    save_txt(os.path.join(FT_DIR, "test_metrics.txt"),
             test_ft, "# métricas en maguey tras fine‑tuning")

    # ========== 3) Consolidation nopal + maguey ==========
    print("\n🌱  Consolidación multiespecie (nopal + maguey)")
    for p in model.parameters(): p.requires_grad = True  # unfreeze total
    optim_cs = torch.optim.AdamW(model.parameters(),
                                 lr=2e-5, weight_decay=1e-5)

    mix_tr = ConcatDataset([n_tr, m_tr])
    mix_vl = ConcatDataset([n_vl, m_vl])
    mix_dl_tr = dl(mix_tr, True); mix_dl_vl = dl(mix_vl, False)

    EPOCHS_CS = 10
    best_cs_iou, hist_cs = 0, []
    for ep in range(1, EPOCHS_CS + 1):
        tr = run_epoch(model, mix_dl_tr, criterion, optim_cs)
        vl = run_epoch(model, mix_dl_vl, criterion)
        hist_cs.append({"epoch": ep, "train": tr, "val": vl})
        if vl["iou"] > best_cs_iou:
            best_cs_iou = vl["iou"]
            torch.save(model.state_dict(),
                       os.path.join(CS_DIR, "unetx_consolidated.pth"))
        print(f"CS {ep:2d}/{EPOCHS_CS} | val IoU:{vl['iou']:.4f}")

    # final test
    model.load_state_dict(torch.load(os.path.join(CS_DIR, "unetx_consolidated.pth"),
                                     map_location=config.DEVICE))
    print("\n Test final nopal:")
    test_np = run_epoch(model, dl(n_te, False), criterion)
    print(" Test final maguey:")
    test_mg = run_epoch(model, te_dl_m, criterion)

    with open(os.path.join(CS_DIR, "test_metrics_final.txt"), "w", encoding="utf-8") as f:
        f.write("# Test nopal\n");  [f.write(f"{k}:{v:.6f}\n") for k, v in test_np.items()]
        f.write("\n# Test maguey\n"); [f.write(f"{k}:{v:.6f}\n") for k, v in test_mg.items()]

if __name__ == "__main__":
    main()