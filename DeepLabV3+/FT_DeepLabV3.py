"""
DeepLabV3+ (ResNet-50) Expansion — Nopal → Maguey → Nopal+Maguey
Phases
0) Pre-test: Direct inference on Maguey dataset.
1) Fine-Tuning: Freeze backbone layers 0-3 and train for 30 epochs on Maguey dataset.
2) Consolidation: Unfreeze all layers and train for Mixed (Nopal+Maguey) dataset.

Author: Arturo Duarte Rangel 
"""
import os, random, time, cv2, numpy as np, torch, csv
from copy import deepcopy
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models.segmentation import deeplabv3_resnet50
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import config

# ─────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ───────────── Dataset ─────────────
class SimpleDataset(Dataset):
    
    def __init__(self, img_dir, msk_dir):
        self.imgs = sorted([os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.endswith(".png")])
        self.msks = sorted([os.path.join(msk_dir,f) for f in os.listdir(msk_dir) if f.endswith(".png")])
        self.t = Compose([Resize(*config.IMAGE_SIZE), ToTensorV2(transpose_mask=True)])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.msks[idx], cv2.IMREAD_GRAYSCALE)
        data = self.t(image=img, mask=msk)
        x = data["image"].float() / 255.0
        y = data["mask"].unsqueeze(0).float() / 255.0

        mean = torch.tensor(config.MEAN)[:, None, None]
        std  = torch.tensor(config.STD )[:, None, None]
        x_norm = (x - mean) / std

        img_raw = cv2.resize(img, config.IMAGE_SIZE[::-1])
        img_raw = torch.from_numpy(img_raw)              # uint8

        return x_norm, y, img_raw

# ───────────── loss and metrics ─────────────
THRESH_MET = 0.35

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=config.BCE_POS_WEIGHT):
        super().__init__()
        self.register_buffer("w", torch.tensor([pos_weight]))
    def forward(self, p, t):
        bce  = nn.functional.binary_cross_entropy_with_logits(p, t, pos_weight=self.w.to(p.device))
        prob = torch.sigmoid(p)
        inter = (prob * t).sum(); union = prob.sum() + t.sum() - inter + 1e-7
        dice = 1 - (2*inter + 1) / (union + 1)
        return bce + dice

@torch.no_grad()
def metrics(log, tgt):
    prob = torch.sigmoid(log)
    pred = (prob > THRESH_MET).float()
    acc  = (pred == tgt).float().mean().item()
    rmse = torch.sqrt(((prob - tgt) ** 2).mean()).item()
    inter = (pred * tgt).sum()
    union = pred.sum() + tgt.sum() - inter + 1e-7
    return acc, rmse, (inter / union).item()

# ───────────── Epochs ─────────────
def run_epoch(model, loader, crit, opt=None):
    model.train(opt is not None)

    losses, accs, rmses, ious, t_imgs = [], [], [], [], []

    for x, y, *_ in (pbar := tqdm(loader, leave=False)):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        tic = time.perf_counter()
        with torch.set_grad_enabled(opt is not None):
            log = model(x)["out"]; loss = crit(log, y)
            if opt:
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if config.DEVICE == "cuda":
            torch.cuda.synchronize()
        t_imgs.append(time.perf_counter() - tic)

        acc, rmse, iou = metrics(log.detach(), y)
        losses.append(loss.item()); accs.append(acc)
        rmses.append(rmse);         ious.append(iou)

        pbar.set_description(f"{'T' if opt else 'V'} L:{loss.item():.3f} IoU:{iou:.3f}")

    agg = {
        "loss"     : np.mean(losses) , "loss_std" : np.std(losses,  ddof=1),
        "acc"      : np.mean(accs)   , "acc_std"  : np.std(accs,    ddof=1),
        "rmse"     : np.mean(rmses)  , "rmse_std" : np.std(rmses,   ddof=1),
        "iou"      : np.mean(ious)   , "iou_std"  : np.std(ious,    ddof=1),
        "t_img"    : np.mean(t_imgs) , "t_img_std": np.std(t_imgs,  ddof=1)
    }
    return agg

# ───────────── Utils ─────────────
def save_txt(path, dct, header=None):
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(header + "\n")
        for k, v in dct.items():
            f.write(f"{k}:{v:.6f}\n")

def dump_csv(dct, path_csv):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["metrics", "mean", "mean_std"])
        for k, v in dct.items():
            if k.endswith("_std"): continue
            wr.writerow([k, v, dct.get(f"{k}_std", 0.0)])

@torch.no_grad()
def save_examples(model, dataset, out_dir, prefix, n=3):
    os.makedirs(out_dir, exist_ok=True)
    for i in random.sample(range(len(dataset)), n):
        x_norm, y, img_raw = dataset[i]
        log = model(x_norm.unsqueeze(0).to(config.DEVICE))["out"].cpu().squeeze()

        img = img_raw.numpy().astype(np.uint8)
        gt  = (y.squeeze().numpy() > 0.5).astype(np.uint8) * 255
        pr  = (torch.sigmoid(log).numpy() > THRESH_MET).astype(np.uint8) * 255
        overlay = img.copy(); overlay[pr == 255] = (255, 0, 0)

        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        for a, t, im in zip(ax, ["Orig", "GT", "Pred", "Overlay"], [img, gt, pr, overlay]):
            a.imshow(im, cmap=None if t == "Orig" else "gray"); a.set_title(t); a.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_{i}.png"), dpi=200); plt.close()

# ───────────── Main ─────────────
def main():
    
    DIR_PRE  = os.path.join(config.FT_DIR_DL, "Pretest")
    DIR_FT   = os.path.join(config.FT_DIR_DL, "FineTune")
    DIR_CONS = os.path.join(config.FT_DIR_DL, "Consolidation")
    for d in (DIR_PRE, DIR_FT, DIR_CONS): os.makedirs(d, exist_ok=True)

    # Datasets
    n_tr = SimpleDataset(config.TRAIN_IMAGES_DIR , config.TRAIN_MASKS_DIR )
    n_va = SimpleDataset(config.VAL_IMAGES_DIR  , config.VAL_MASKS_DIR  )
    n_te = SimpleDataset(config.TEST_IMAGES_DIR , config.TEST_MASKS_DIR )

    m_tr = SimpleDataset(config.TRAIN_IMAGES_DIRM, config.TRAIN_MASKS_DIRM)
    m_va = SimpleDataset(config.VAL_IMAGES_DIRM  , config.VAL_MASKS_DIRM)
    m_te = SimpleDataset(config.TEST_IMAGES_DIRM , config.TEST_MASKS_DIRM)

    def make_dl(ds, shuffle=False):
        return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle,
                          num_workers=config.NUM_WORKERS, pin_memory=True)
    dl = {
        "n_tr"  : make_dl(n_tr, True),  "n_va"  : make_dl(n_va), "n_te": make_dl(n_te),
        "m_tr"  : make_dl(m_tr, True),  "m_va"  : make_dl(m_va), "m_te": make_dl(m_te),
        "mix_tr": make_dl(ConcatDataset([n_tr, m_tr]), True),
        "mix_va": make_dl(ConcatDataset([n_va, m_va]))
    }

    # Model base (nopal)
    model = deeplabv3_resnet50(weights=None, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    ckpt = os.path.join(config.CHECKPOINT_DIR, "deeplab_best.pth")
    assert os.path.exists(ckpt), f"Checkpoint not found: {ckpt}"
    model.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
    criterion = BCEDiceLoss()

    # ═════ FASE 0 – PRE-TEST ═════
    print("\n Pre-test maguey")
    pre = run_epoch(model, dl["m_te"], criterion)
    save_txt(os.path.join(DIR_PRE, "metrics_pretest.txt"),
             pre, header="#direct inference (modelo nopal)")
    dump_csv(pre, os.path.join(DIR_PRE, "metrics_pretest.csv"))
    save_examples(model, m_te, DIR_PRE, "pre", n=3)

    # ═════ FASE 1 – FINE-TUNING ═════
    print("\n Fine-Tuning en maguey")
    for name, p in model.named_parameters():
        if name.startswith(("backbone.body.layer0",
                            "backbone.body.layer1",
                            "backbone.body.layer2",
                            "backbone.body.layer3")):
            p.requires_grad = False
    opt_ft = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=1e-4, weight_decay=1e-5)

    BEST_W, best_iou = None, 0.0; EPOCHS_FT = 30
    for ep in range(1, EPOCHS_FT + 1):
        tr = run_epoch(model, dl["m_tr"], criterion, opt_ft)
        vl = run_epoch(model, dl["m_va"], criterion)
        print(f"FT {ep:2d}/{EPOCHS_FT} | Train IoU:{tr['iou']:.4f} | Val IoU:{vl['iou']:.4f}")
        if vl["iou"] > best_iou:
            best_iou, BEST_W = vl["iou"], deepcopy(model.state_dict())
    torch.save(BEST_W, os.path.join(DIR_FT, "deeplab_finetuned.pth"))

    # Post-FT métricas + visuales
    model.load_state_dict(BEST_W)
    ft_test = run_epoch(model, dl["m_te"], criterion)
    save_txt(os.path.join(DIR_FT, "metrics_finetune.txt"),
             ft_test, header="# métricas en maguey tras fine-tuning")
    dump_csv(ft_test, os.path.join(DIR_FT, "metrics_finetune.csv"))
    save_examples(model, m_te, DIR_FT, "ft", n=5)

    # ═════ FASE 2 – CONSOLIDACIÓN ═════
    print("\n nopal + maguey")
    for p in model.parameters(): p.requires_grad = True
    opt_cons = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    EPOCHS_CONS = 10
    for ep in range(1, EPOCHS_CONS + 1):
        run_epoch(model, dl["mix_tr"], criterion, opt_cons)
        vl = run_epoch(model, dl["mix_va"], criterion)
        print(f"CONS {ep}/{EPOCHS_CONS} | Mix Val IoU:{vl['iou']:.4f}")
    torch.save(model.state_dict(), os.path.join(DIR_CONS, "deeplab_final.pth"))

    # Métricas finales
    final_m = run_epoch(model, dl["m_te"], criterion)
    final_n = run_epoch(model, dl["n_te"], criterion)
    save_txt(os.path.join(DIR_CONS, "metrics_maguey.txt"),
             final_m, header="# final metrics maguey")
    save_txt(os.path.join(DIR_CONS, "metrics_nopal.txt"),
             final_n, header="# métricas finales en nopal")
    dump_csv(final_m, os.path.join(DIR_CONS, "metrics_maguey.csv"))
    dump_csv(final_n, os.path.join(DIR_CONS, "metrics_nopal.csv"))
    save_examples(model, m_te, DIR_CONS, "mag_fin", n=3)
    save_examples(model, n_tr, DIR_CONS, "nop_fin", n=3)

    print("\n Complete → deeplab_final.pth")

if __name__ == "__main__":
    main()