import os, random, time, cv2, numpy as np, torch, csv
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from albumentations import (Compose, HorizontalFlip, VerticalFlip,
                            Rotate, RandomBrightnessContrast, Resize)
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import config                                                

THRESH_VIS = 0.6
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ───────── Dataset ─────────
class BinarySegDataset(Dataset):
    def __init__(self, img_dir, msk_dir, augment=False):
        self.imgs = sorted([os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.endswith(".png")])
        self.msks = sorted([os.path.join(msk_dir,f) for f in os.listdir(msk_dir) if f.endswith(".png")])
        self.aug = augment
        self.t_train = Compose([
            HorizontalFlip(p=0.5), VerticalFlip(p=0.5),
            Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            RandomBrightnessContrast(0.2,0.2,p=0.3),
            Resize(*config.IMAGE_SIZE), ToTensorV2(transpose_mask=True)])
        self.t_plain = Compose([Resize(*config.IMAGE_SIZE), ToTensorV2(transpose_mask=True)])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.msks[idx], cv2.IMREAD_GRAYSCALE)
        tf = self.t_train if self.aug else self.t_plain
        data = tf(image=img, mask=mask)
        x = data["image"].float()
        y = data["mask"].unsqueeze(0).float() / 255.
        mean = torch.tensor(config.MEAN)[:, None, None]
        std  = torch.tensor(config.STD)[:,  None, None]
        x = (x - mean) / std
        return x, y, img_path                               

# ───────── Model and loss ─────────
def get_model():
    return deeplabv3_resnet50(weights=None, aux_loss=None, num_classes=config.NUM_CLASSES)

def dice_coeff(logits, target, smooth=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2 * inter + smooth) / (union + smooth)).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=config.BCE_POS_WEIGHT):
        super().__init__()
        self.register_buffer("pos_w", torch.tensor([pos_weight]))
    def forward(self, p, t):
        bce = nn.functional.binary_cross_entropy_with_logits(p, t, pos_weight=self.pos_w.to(p.device))
        return bce + (1 - dice_coeff(p, t))

@torch.no_grad()
def metrics(log, tgt, t=0.5, eps=1e-7):
    prob = torch.sigmoid(log)
    pred = (prob > t).float()
    acc  = (pred == tgt).float().mean().item()
    rmse = torch.sqrt(((prob - tgt) ** 2).mean()).item()
    inter = (pred * tgt).sum()
    union = pred.sum() + tgt.sum() - inter + eps
    return acc, rmse, (inter / union).item()

# ───────── Epochs ─────────
def run_epoch(model, loader, crit, opt=None):
    
    model.train(opt is not None)

    losses, accs, rmses, ious, t_imgs = [], [], [], [], []

    for x, y, *_ in (pbar := tqdm(loader, leave=False,
                                  desc="train" if opt else "eval")):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        tic = time.perf_counter()
        with torch.set_grad_enabled(opt is not None):
            log = model(x)["out"]
            loss = crit(log, y)
            if opt:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
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

# ───────── Visualization ─────────
def save_panel(img_path, mask_tgt, logit, path):
    img_vis = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_vis = cv2.resize(img_vis, config.IMAGE_SIZE)
    gt  = (mask_tgt.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    pr  = (torch.sigmoid(logit).cpu().squeeze().numpy() > THRESH_VIS).astype(np.uint8) * 255
    overlay = img_vis.copy(); overlay[pr == 255] = (255, 0, 0)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    for a, t, im in zip(ax, ["Original", "GT", "Pred", "Overlay"], [img_vis, gt, pr, overlay]):
        a.imshow(im, cmap=None if t == "Original" else "gray"); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def dump_csv(metrics_dict, path_csv):
    """Guarda CSV con cabecera metrica,media,desv_std."""
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metrica", "media", "desv_std"])
        for k, v in metrics_dict.items():
            if k.endswith("_std"):                 
                continue
            writer.writerow([k, v, metrics_dict.get(f"{k}_std", 0.0)])

# ───────── Main ─────────
def main():
    tr_ds = BinarySegDataset(config.TRAIN_IMAGES_DIR, config.TRAIN_MASKS_DIR, True)
    vl_ds = BinarySegDataset(config.VAL_IMAGES_DIR , config.VAL_MASKS_DIR , False)
    te_ds = BinarySegDataset(config.TEST_IMAGES_DIR, config.TEST_MASKS_DIR, False)

    tr_dl = DataLoader(tr_ds, batch_size=config.BATCH_SIZE, shuffle=True ,
                       num_workers=config.NUM_WORKERS, pin_memory=True)
    vl_dl = DataLoader(vl_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                       num_workers=config.NUM_WORKERS, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                       num_workers=config.NUM_WORKERS, pin_memory=True)

    model = get_model().to(config.DEVICE)
    crit  = BCEDiceLoss()
    opt   = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                        weight_decay=config.WEIGHT_DECAY)
    sch   = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    hist, best = [], 0.0
    for ep in range(1, config.NUM_EPOCHS + 1):
        tr = run_epoch(model, tr_dl, crit, opt)
        vl = run_epoch(model, vl_dl, crit)
        hist.append({"epoch": ep, "train": tr, "val": vl})

        if vl["iou"] > best:
            best = vl["iou"]
            torch.save(model.state_dict(),
                       os.path.join(config.CHECKPOINT_DIR, "deeplab_best.pth"))
        sch.step(vl["iou"])

        if ep % 10 == 0:
            print(f"Ep {ep}/{config.NUM_EPOCHS} | Train IoU:{tr['iou']:.3f} | Val IoU:{vl['iou']:.3f}")

    # ----- TRAIN -----
    model.eval()
    for i in random.sample(range(len(tr_ds)), 5):
        x, y, img_path = tr_ds[i]
        with torch.no_grad():
            log = model(x.unsqueeze(0).to(config.DEVICE))["out"].cpu().squeeze(0)
        save_panel(img_path, y, log, os.path.join(config.CHECKPOINT_DIR, f"train_vis_{i}.png"))

    # ----- TEST -----
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "deeplab_best.pth"),
                                     map_location=config.DEVICE))
    test = run_epoch(model, te_dl, crit)

    # TXT
    with open(os.path.join(config.TEST_OUT_DIR, "test_metrics.txt"), "w") as f:
        for k, v in test.items():
            f.write(f"{k}:{v:.6f}\n")
    # CSV
    dump_csv(test, os.path.join(config.TEST_OUT_DIR, "test_metrics.csv"))

    for i in random.sample(range(len(te_ds)), 5):
        x, y, img_path = te_ds[i]
        with torch.no_grad():
            log = model(x.unsqueeze(0).to(config.DEVICE))["out"].cpu().squeeze(0)
        save_panel(img_path, y, log, os.path.join(config.TEST_OUT_DIR, f"test_vis_{i}.png"))

if __name__ == "__main__":
    main()