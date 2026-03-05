# -*- coding: utf-8 -*-
"""
FINE TUNNING 

Segnet Expansion — Nopal → Maguey → Nopal+Maguey
Phases
0) Pre-test: Direct inference on Maguey dataset.
1) Fine-Tuning: Freeze backbone layers 0-3 and train for 30 epochs on Maguey dataset.
2) Consolidation: Unfreeze all layers and train on the Mixed (Nopal+Maguey) dataset.
"""

import os, random, time, json
from copy import deepcopy

import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

import config
from Seg_Net_Model_Metricas import SegNet, metrics, save_plot

# ───────────────────────── Dataset ────────────────────────────
class SimpleDataset(Dataset):
    
    def __init__(self, img_dir, msk_dir):
        self.imgs = sorted([os.path.join(img_dir, f)
                            for f in os.listdir(img_dir) if f.endswith(".png")])
        self.msks = sorted([os.path.join(msk_dir, f)
                            for f in os.listdir(msk_dir) if f.endswith(".png")])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.msks[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, config.IMAGE_SIZE[::-1]) / 255.0
        msk = (cv2.resize(msk, config.IMAGE_SIZE[::-1]) / 255.0 > 0.5).astype(np.float32)
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        msk = torch.tensor(msk, dtype=torch.float32).unsqueeze(0)
        return img, msk

# ───────────────────────── épochs ──────────────────────────────
def run_epoch(model, loader, criterion, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    agg = {"loss":0,"acc":0,"rmse":0,"iou":0}; t_tot=0; n_img=0
    for imgs, msks in (pbar:=tqdm(loader, leave=False)):
        imgs, msks = imgs.to(config.DEVICE), msks.to(config.DEVICE)
        tic = time.perf_counter()
        with torch.set_grad_enabled(train):
            outs = model(imgs)
            loss = criterion(outs, msks)
            if train:
                optim.zero_grad(); loss.backward(); optim.step()
        torch.cuda.synchronize() if config.DEVICE=="cuda" else None
        t_tot += time.perf_counter() - tic; n_img += imgs.size(0)
        acc, rmse, iou = metrics(outs.detach(), msks)
        for k,v in zip(["loss","acc","rmse","iou"], [loss.item(), acc, rmse, iou]):
            agg[k] += v
        pbar.set_description(f"L:{loss.item():.3f} IoU:{iou:.3f}")
    n = len(loader); agg = {k:v/n for k,v in agg.items()}; agg["t_img"] = t_tot / n_img
    return agg

# ───────────────────────── Utilies ──────────────────────────
def save_txt(path, dct, header=None):
    with open(path, "w", encoding="utf-8") as f:
        if header: f.write(header + "\n")
        for k, v in dct.items(): f.write(f"{k}:{v:.6f}\n")

def save_examples(model, dataset, out_dir, prefix, n=3):
    os.makedirs(out_dir, exist_ok=True)
    for i in random.sample(range(len(dataset)), n):
        img, msk = dataset[i]
        with torch.no_grad():
            pr = model(img.unsqueeze(0).to(config.DEVICE)).cpu().squeeze().numpy()
        save_plot(
            (img.permute(1,2,0).numpy()*255).astype(np.uint8),          # RGB
            (msk.squeeze().numpy()*255).astype(np.uint8),               # GT mask
            pr,                                                         # pred flotante
            os.path.join(out_dir, f"{prefix}_{i}.png")
        )

# ─────────────────────────────── Main ─────────────────────────────────
def main():
    # ──── out ────
    DIR_PRE  = os.path.join(config.FT_DIR, "Pretest")
    DIR_FT   = os.path.join(config.FT_DIR, "FineTune")
    DIR_CONS = os.path.join(config.FT_DIR, "Consolidation")
    for d in (DIR_PRE, DIR_FT, DIR_CONS): os.makedirs(d, exist_ok=True)

    # ──── dataset ────
    n_tr = SimpleDataset(config.TRAIN_IMAGES_DIR,  config.TRAIN_MASKS_DIR)
    n_va = SimpleDataset(config.VAL_IMAGES_DIR,    config.VAL_MASKS_DIR)
    n_te = SimpleDataset(config.TEST_IMAGES_DIR,   config.TEST_MASKS_DIR)

    m_tr = SimpleDataset(config.TRAIN_IMAGES_DIRM, config.TRAIN_MASKS_DIRM)
    m_va = SimpleDataset(config.VAL_IMAGES_DIRM,   config.VAL_MASKS_DIRM)
    m_te = SimpleDataset(config.TEST_IMAGES_DIRM,  config.TEST_MASKS_DIRM)

    # ──── dataloaders ────
    def make_dl(ds, shuffle=False):
        return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle,
                          num_workers=config.NUM_WORKERS, pin_memory=True)
    dl = {
        "n_tr":make_dl(n_tr, True),  "n_va":make_dl(n_va),
        "m_tr":make_dl(m_tr, True),  "m_va":make_dl(m_va), "m_te":make_dl(m_te),
        "mix_tr":make_dl(ConcatDataset([n_tr, m_tr]), True),
        "mix_va":make_dl(ConcatDataset([n_va, m_va]))
    }

    # ──── model ────
    model = SegNet(config.NUM_CLASSES).to(config.DEVICE)
    ckpt = os.path.join(config.CHECKPOINT_DIR, "5/segnet_best.pth")
    assert os.path.exists(ckpt), f"Checkpoint no encontrado: {ckpt}"
    model.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
    criterion = nn.BCELoss()

    # ═════════════════════════ FASE 0 – PRE-TEST ════════════════════════
    print("\n Pre-test en maguey")
    pre = run_epoch(model, dl["m_te"], criterion)
    save_txt(os.path.join(DIR_PRE, "metrics_pretest.txt"),
             pre, header="# métricas inferencia directa (modelo nopal)")
    save_examples(model, m_te, DIR_PRE, "pre", n=3)

    # ═════════════════════════ FASE 1 – FINE-TUNING ═════════════════════
    print("\n Fine-Tuning en maguey")
    # congelar enc1–enc3
    for name, prm in model.named_parameters():
        if name.split('.')[0] in {"enc1","enc2","enc3"}: prm.requires_grad = False
    opt_ft = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=1e-4, weight_decay=1e-5)
    BEST_W = None; best_iou = 0
    EPOCHS_FT = 30
    for epoch in range(1, EPOCHS_FT+1):
        tr = run_epoch(model, dl["m_tr"], criterion, opt_ft)
        vl = run_epoch(model, dl["m_va"], criterion)
        print(f"FT {epoch:2d}/{EPOCHS_FT} | Train IoU:{tr['iou']:.4f} | Val IoU:{vl['iou']:.4f}")
        if vl["iou"] > best_iou:
            best_iou, BEST_W = vl["iou"], deepcopy(model.state_dict())
    # guardar mejor modelo FT
    torch.save(BEST_W, os.path.join(DIR_FT, "segnet_finetuned.pth"))
    # métricas y ejemplos post-FT
    model.load_state_dict(BEST_W)
    ft_test = run_epoch(model, dl["m_te"], criterion)
    save_txt(os.path.join(DIR_FT, "metrics_finetune.txt"),
             ft_test, header="# métricas en maguey tras fine-tuning")
    save_examples(model, m_te, DIR_FT, "ft", n=5)

    # ════════════════════════ FASE 2 – CONS ═══════════════════
    print("\n Consolidación nopal + maguey")
    # desbloquear todo
    for prm in model.parameters(): prm.requires_grad = True
    opt_cons = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    EPOCHS_CONS = 10
    for epoch in range(1, EPOCHS_CONS+1):
        run_epoch(model, dl["mix_tr"], criterion, opt_cons)
        vl = run_epoch(model, dl["mix_va"], criterion)
        print(f"CONS {epoch}/{EPOCHS_CONS} | Mix Val IoU:{vl['iou']:.4f}")
    torch.save(model.state_dict(), os.path.join(DIR_CONS, "segnet_final.pth"))

    # ──── metrics ────
    final_m = run_epoch(model, dl["m_te"], criterion)
    final_n = run_epoch(model, dl["n_tr"], criterion)
    save_txt(os.path.join(DIR_CONS, "metrics_maguey.txt"),
             final_m, header="# métricas finales en maguey")
    save_txt(os.path.join(DIR_CONS, "metrics_nopal.txt"),
             final_n, header="# métricas finales en nopal")
    save_examples(model, m_te, DIR_CONS, "mag_fin", n=3)
    save_examples(model, n_tr, DIR_CONS, "nop_fin", n=3)

    print("\n Completed → segnet_final.pth")

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    main()