import os, random, time
import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from UnetStyleXception import config_xception as config


# ═════════════════ Dataset ════════════════
class NopalDataset(Dataset):
    def __init__(self, img_dir, msk_dir, augment=False):
        self.ip = sorted([os.path.join(img_dir, f)
                          for f in os.listdir(img_dir) if f.endswith(".png")])
        self.mp = sorted([os.path.join(msk_dir, f)
                          for f in os.listdir(msk_dir) if f.endswith(".png")])
        self.augment = augment
        self.tf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.ColorJitter(0.1, 0.1, 0.1, 0.05, p=0.3),
        ])

    def __len__(self): return len(self.ip)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.ip[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.mp[idx], cv2.IMREAD_GRAYSCALE)
        if self.augment:
            aug = self.tf(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]
        img = cv2.resize(img, config.IMAGE_SIZE[::-1]) / 255.0
        msk = cv2.resize(msk, config.IMAGE_SIZE[::-1]) / 255.0
        msk = (msk > 0.5).astype(np.float32)
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        msk = torch.tensor(msk, dtype=torch.float32).unsqueeze(0)
        return img, msk

# ═════════════════ UNet‑Style‑Xception ════════════════
def sep_conv(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, in_c, k, s, p, groups=in_c, bias=False),
        nn.BatchNorm2d(in_c), nn.ReLU(inplace=True),
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = sep_conv(in_c, out_c)
        self.conv2 = sep_conv(out_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c1 = sep_conv(in_c + skip_c, out_c)
        self.c2 = sep_conv(out_c, out_c)

    def forward(self, x, skip):
        x = self.up(x); x = torch.cat([x, skip], 1)
        x = self.c1(x); x = self.c2(x)
        return x

class UNetXception(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        filters = [32, 64, 128, 256, 512]
        self.e1 = EncoderBlock(3,        filters[0])
        self.e2 = EncoderBlock(filters[0], filters[1])
        self.e3 = EncoderBlock(filters[1], filters[2])
        self.e4 = EncoderBlock(filters[2], filters[3])
        self.bottom = sep_conv(filters[3], filters[4])
        self.d4 = DecoderBlock(filters[4], filters[3], filters[3])
        self.d3 = DecoderBlock(filters[3], filters[2], filters[2])
        self.d2 = DecoderBlock(filters[2], filters[1], filters[1])
        self.d1 = DecoderBlock(filters[1], filters[0], filters[0])
        self.out_conv = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, x):
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        x = self.bottom(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        return self.out_conv(x)

# ═════════════════ Combined loss ════════════════
class ComboLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dw  = dice_weight

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum((2, 3))
        union = probs.sum((2, 3)) + target.sum((2, 3)) + 1e-7
        dice = 1 - (2 * inter / union).mean()
        return bce + self.dw * dice

# ═════════════════ Métrics ════════════════
def metrics(logits, mask):
    probs = torch.sigmoid(logits)
    pred_bin = (probs > 0.5).float()
    acc  = (pred_bin == mask).float().mean().item()
    rmse = torch.sqrt(((probs - mask) ** 2).mean()).item()
    inter = (pred_bin * mask).sum()
    union = pred_bin.sum() + mask.sum() - inter + 1e-7
    iou = (inter / union).item()
    return acc, rmse, iou

# ═════════════════ Epoch loop ════════════════
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

    n = len(loader); agg = {k: v / n for k, v in agg.items()}
    agg["t_img"] = t_tot / n_img
    return agg

# ═════════════════ Visualization ════════════════
def save_plot(img, msk, logits, path):
    probs = 1 / (1 + np.exp(-logits))
    pred_bin = (probs > 0.5).astype(np.uint8) * 255
    overlay = img.copy(); overlay[pred_bin == 255] = (255, 0, 0)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(img); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(msk, cmap="gray"); ax[1].set_title("GT"); ax[1].axis("off")
    ax[2].imshow(pred_bin, cmap="gray"); ax[2].set_title("Pred"); ax[2].axis("off")
    ax[3].imshow(overlay); ax[3].set_title("Overlay"); ax[3].axis("off")
    plt.tight_layout(); plt.savefig(path); plt.close()

# ═════════════════ Main ════════════════
def main():
    # ─── Data
    tr_ds = NopalDataset(config.TRAIN_IMAGES_DIR, config.TRAIN_MASKS_DIR, augment=True)
    vl_ds = NopalDataset(config.VAL_IMAGES_DIR,   config.VAL_MASKS_DIR)
    te_ds = NopalDataset(config.TEST_IMAGES_DIR,  config.TEST_MASKS_DIR)

    def make_dl(ds, shf):
        return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shf,
                          num_workers=config.NUM_WORKERS, pin_memory=True)

    tr_dl = make_dl(tr_ds, True); vl_dl = make_dl(vl_ds, False); te_dl = make_dl(te_ds, False)

    # ─── Model, loss, and optimizer
    model = UNetXception(config.NUM_CLASSES).to(config.DEVICE)
    criterion = ComboLoss(dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    # ─── Training
    best_iou, patience, PATIENCE_MAX = 0.0, 0, 15
    history = []

    print(f"\n🧠  Device: {config.DEVICE.upper()} | Epochs: {config.NUM_EPOCHS}")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        tr = run_epoch(model, tr_dl, criterion, optimizer)
        vl = run_epoch(model, vl_dl, criterion)

        history.append({"epoch": epoch, "train": tr, "val": vl})

        scheduler.step(vl["iou"])
        if vl["iou"] > best_iou:
            best_iou = vl["iou"]; patience = 0
            torch.save(model.state_dict(),
                       os.path.join(config.CHECKPOINT_DIR, "unetx_best.pth"))
        else:
            patience += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
                  f"Train IoU:{tr['iou']:.4f} | Val IoU:{vl['iou']:.4f} | "
                  f"LR:{optimizer.param_groups[0]['lr']:.1e}")

        if patience >= PATIENCE_MAX:
            print(f"⏹ Early‑Stopping (sin mejora en {PATIENCE_MAX} épocas)")
            break

    # ─── Save full metrics
    def write_txt(path, hist):
        keys = ["loss", "acc", "rmse", "iou", "t_img"]
        with open(path, "w", encoding="utf-8") as f:
            f.write("# epoch " + " ".join([f"train_{k}" for k in keys]) +
                    " " + " ".join([f"val_{k}" for k in keys]) + "\n")
            for h in hist:
                vals = [h["train"][k] for k in keys] + [h["val"][k] for k in keys]
                f.write(f"{h['epoch']:3d} " + " ".join(f"{v:.6f}" for v in vals) + "\n")
            f.write("\n# media ± std\n")
            for split in ("train", "val"):
                arr = lambda k: np.array([h[split][k] for h in hist])
                f.write(f"{split.capitalize():5s}: " + " | ".join(
                    f"{k}:{arr(k).mean():.4f}±{arr(k).std():.4f}" for k in keys) + "\n")

    write_txt(os.path.join(config.CHECKPOINT_DIR, "train_metrics_unetx.txt"), history)

    # ─── Training visualizations
    for i in random.sample(range(len(tr_ds)), 5):
        img, msk = tr_ds[i]
        with torch.no_grad():
            logit = model(img.unsqueeze(0).to(config.DEVICE)).cpu().squeeze().numpy()
        save_plot((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                  (msk.squeeze().numpy() * 255).astype(np.uint8),
                  logit,
                  os.path.join(config.CHECKPOINT_DIR, f"train_vis_unetx_{i}.png"))

    # ─── Final test
    test = run_epoch(model, te_dl, criterion)
    print("\n📊  Test:", test)

    with open(os.path.join(config.TEST_OUT_DIR, "test_metrics_unetx.txt"),
              "w", encoding="utf-8") as f:
        for k, v in test.items():
            f.write(f"{k}:{v:.6f}\n")

    for i in random.sample(range(len(te_ds)), 5):
        img, msk = te_ds[i]
        with torch.no_grad():
            logit = model(img.unsqueeze(0).to(config.DEVICE)).cpu().squeeze().numpy()
        save_plot((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                  (msk.squeeze().numpy() * 255).astype(np.uint8),
                  logit,
                  os.path.join(config.TEST_OUT_DIR, f"test_vis_unetx_{i}.png"))

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    main()