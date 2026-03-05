
"""
SegNet
Author: Arturo Duarte Rangel 
"""

import os, random, time, json
import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm
import config

# ───────────────────────────────────────
THRESH_BIN   = 0.45   # umbral
PATIENCE_ES  = 8      # early-stopping
ROT_ANGLE    = 15     # ±15°
DROPOUT_P    = 0.30   # dropout decoder

# ───────────────────────── Dataset ──────────────────────
class NopalDataset(Dataset):
    def __init__(self, img_dir, msk_dir, augment=False):
        self.imgs = sorted([os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.endswith(".png")])
        self.msks = sorted([os.path.join(msk_dir,f) for f in os.listdir(msk_dir) if f.endswith(".png")])
        self.augment = augment
    def __len__(self): return len(self.imgs)
    def _augment(self, img, msk):
        # Flip H/V
        if random.random() < .5:
            img = TF.hflip(img); msk = TF.hflip(msk)
        if random.random() < .5:
            img = TF.vflip(img); msk = TF.vflip(msk)
        # Rotación
        angle = random.uniform(-ROT_ANGLE, ROT_ANGLE)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        msk = TF.rotate(msk, angle, interpolation=TF.InterpolationMode.NEAREST)
        # Jitter brillo-contraste-saturación
        img = TF.adjust_brightness(img, random.uniform(0.8,1.2))
        img = TF.adjust_contrast(img,  random.uniform(0.8,1.2))
        img = TF.adjust_saturation(img,random.uniform(0.8,1.2))
        return img, msk
    def __getitem__(self, i):
        # --- lectura ---
        img = cv2.cvtColor(cv2.imread(self.imgs[i]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.msks[i], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, config.IMAGE_SIZE[::-1])
        msk = cv2.resize(msk, config.IMAGE_SIZE[::-1])
        # --- PIL → torchvision tensor para usar TF augment —
        img = TF.to_pil_image(img)
        msk = TF.to_pil_image(msk)
        if self.augment: img,msk = self._augment(img, msk)
        img = TF.to_tensor(img)                      # [0,1]
        msk = TF.to_tensor(msk).float()
        msk = (msk > .5).float()                     # binariza
        return img, msk

# ────────────────── conv-BN-ReLU(+Dropout) ────────────────────
def conv_bn_relu(in_c, out_c, k=3, drop=False):
    layers = [nn.Conv2d(in_c,out_c,k,padding=k//2),
              nn.BatchNorm2d(out_c),
              nn.ReLU(inplace=True)]
    if drop: layers.append(nn.Dropout2d(DROPOUT_P))
    return nn.Sequential(*layers)

# ─────────────────────────── Modelo SegNet ───────────────────────────
class SegNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.enc1 = nn.Sequential(conv_bn_relu(3,64), conv_bn_relu(64,64))
        self.enc2 = nn.Sequential(conv_bn_relu(64,128), conv_bn_relu(128,128))
        self.enc3 = nn.Sequential(conv_bn_relu(128,256), conv_bn_relu(256,256), conv_bn_relu(256,256))
        self.enc4 = nn.Sequential(conv_bn_relu(256,512), conv_bn_relu(512,512), conv_bn_relu(512,512))
        self.enc5 = nn.Sequential(conv_bn_relu(512,512), conv_bn_relu(512,512), conv_bn_relu(512,512))
        self.pool, self.unpool = nn.MaxPool2d(2,2,return_indices=True), nn.MaxUnpool2d(2,2)
        self.dec5 = nn.Sequential(conv_bn_relu(512,512,drop=True), conv_bn_relu(512,512,drop=True),
                                  conv_bn_relu(512,512,drop=True))
        self.dec4 = nn.Sequential(conv_bn_relu(512,512,drop=True), conv_bn_relu(512,512,drop=True),
                                  conv_bn_relu(512,256,drop=True))
        self.dec3 = nn.Sequential(conv_bn_relu(256,256,drop=True), conv_bn_relu(256,256,drop=True),
                                  conv_bn_relu(256,128,drop=True))
        self.dec2 = nn.Sequential(conv_bn_relu(128,128,drop=True), conv_bn_relu(128,64,drop=True))
        self.dec1 = nn.Sequential(conv_bn_relu(64,64,drop=True), nn.Conv2d(64,n_classes,1))
    def forward(self,x):
        x1 = self.enc1(x); s1,idx1 = self.pool(x1)
        x2 = self.enc2(s1); s2,idx2 = self.pool(x2)
        x3 = self.enc3(s2); s3,idx3 = self.pool(x3)
        x4 = self.enc4(s3); s4,idx4 = self.pool(x4)
        x5 = self.enc5(s4); s5,idx5 = self.pool(x5)
        d5 = self.unpool(s5,idx5); d5 = self.dec5(d5)
        d4 = self.unpool(d5,idx4); d4 = self.dec4(d4)
        d3 = self.unpool(d4,idx3); d3 = self.dec3(d3)
        d2 = self.unpool(d3,idx2); d2 = self.dec2(d2)
        d1 = self.unpool(d2,idx1); d1 = self.dec1(d1)
        return torch.sigmoid(d1)

# ─────────────── loss BCE + Dice ───────────────
class BCEDice(nn.Module):
    def __init__(self): super().__init__(); self.bce = nn.BCELoss()
    def forward(self,pred,mask):
        dice = 1 - (2*(pred*mask).sum()+1)/(pred.sum()+mask.sum()+1)
        return 0.5*self.bce(pred,mask)+0.5*dice

# ─────────────── Metrics ───────────────
def compute_metrics(pred, mask):
    pred_bin = (pred > THRESH_BIN).float()
    acc  = (pred_bin==mask).float().mean().item()
    rmse = torch.sqrt(((pred-mask)**2).mean()).item()
    inter = (pred_bin*mask).sum()
    union = pred_bin.sum() + mask.sum() - inter + 1e-7
    iou  = (inter/union).item()
    return acc, rmse, iou

# ─────────────── epochs ───────────────
def run_epoch(model, loader, criterion, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    stats = {"loss":0,"acc":0,"rmse":0,"iou":0}; t_tot=0; n_img=0
    for img,msk in (pbar:=tqdm(loader, leave=False)):
        img,msk = img.to(config.DEVICE), msk.to(config.DEVICE)
        tic = time.perf_counter()
        with torch.set_grad_enabled(train):
            out = model(img)
            loss = criterion(out, msk)
            if train:
                optim.zero_grad(); loss.backward(); optim.step()
        torch.cuda.synchronize() if config.DEVICE=="cuda" else None
        t_tot += time.perf_counter()-tic; n_img += img.size(0)
        acc,rmse,iou = compute_metrics(out.detach(), msk)
        for k,v in zip(stats.keys(), [loss.item(), acc, rmse, iou]): stats[k]+=v
        pbar.set_description(f"L:{loss.item():.3f} IoU:{iou:.3f}")
    n=len(loader); stats={k:v/n for k,v in stats.items()}; stats["t_img"]=t_tot/n_img
    return stats

# ──────────────────────────────
def save_plot(img, msk, pred, path):
    pred_bin = (pred>THRESH_BIN).astype(np.uint8)*255
    overlay = img.copy(); overlay[pred_bin==255]=(255,0,0)
    fig,ax = plt.subplots(1,4,figsize=(12,3))
    for a in ax: a.axis("off")
    ax[0].imshow(img); ax[0].set_title("Original")
    ax[1].imshow(msk,cmap="gray"); ax[1].set_title("GT")
    ax[2].imshow(pred_bin,cmap="gray"); ax[2].set_title("Pred")
    ax[3].imshow(overlay); ax[3].set_title("Overlay")
    plt.tight_layout(); plt.savefig(path); plt.close()

# ─────────────── Main ───────────────
def main():
    # seed
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark=False

    tr_ds = NopalDataset(config.TRAIN_IMAGES_DIR, config.TRAIN_MASKS_DIR, augment=True)
    vl_ds = NopalDataset(config.VAL_IMAGES_DIR,   config.VAL_MASKS_DIR,   augment=False)
    te_ds = NopalDataset(config.TEST_IMAGES_DIR,  config.TEST_MASKS_DIR,  augment=False)
    tr_dl = DataLoader(tr_ds,batch_size=config.BATCH_SIZE,shuffle=True,
                       num_workers=config.NUM_WORKERS,pin_memory=True)
    vl_dl = DataLoader(vl_ds,batch_size=config.BATCH_SIZE,shuffle=False,
                       num_workers=config.NUM_WORKERS,pin_memory=True)
    te_dl = DataLoader(te_ds,batch_size=config.BATCH_SIZE,shuffle=False,
                       num_workers=config.NUM_WORKERS,pin_memory=True)

    model = SegNet(config.NUM_CLASSES).to(config.DEVICE)
    criterion = BCEDice()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=4, verbose=True)

    history=[]; best_iou=0; patience=0
    print(f"\n Entrenando en {config.DEVICE.upper()} …")
    for epoch in range(1, config.NUM_EPOCHS+1):
        tr = run_epoch(model, tr_dl, criterion, optimizer)
        vl = run_epoch(model, vl_dl, criterion)
        history.append({"epoch":epoch,"train":tr,"val":vl})
        if epoch%10==0:
            print(f"Epoch {epoch:3d} | Train IoU:{tr['iou']:.4f} | Val IoU:{vl['iou']:.4f}")
        scheduler.step(vl["loss"])

        if vl["iou"] > best_iou:
            best_iou = vl["iou"]; patience=0
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR,"segnet_best.pth"))
        else:
            patience += 1
        if patience > PATIENCE_ES:
            print(" Early-stopping ejecutado."); break

    # ── métricas train/val a .txt ──
    keys=["loss","acc","rmse","iou","t_img"]
    with open(os.path.join(config.CHECKPOINT_DIR,"train_metrics.txt"),"w",encoding="utf-8") as f:
        f.write("# epoch "+" ".join([f"train_{k}" for k in keys])+" "+
                " ".join([f"val_{k}" for k in keys])+"\n")
        for h in history:
            vals=[h["train"][k] for k in keys]+[h["val"][k] for k in keys]
            f.write(f"{h['epoch']:3d} "+" ".join([f"{v:.6f}" for v in vals])+"\n")
        f.write("\n# media ± std\n")
        for split in ["train","val"]:
            arr = lambda k: np.array([h[split][k] for h in history])
            f.write(f"{split}: "+" | ".join(
                [f"{k}:{arr(k).mean():.4f}±{arr(k).std():.4f}" for k in keys])+"\n")

    # ── visualizaciones train (5) ──
    for i in random.sample(range(len(tr_ds)),5):
        img,msk = tr_ds[i]
        with torch.no_grad():
            pr = model(img.unsqueeze(0).to(config.DEVICE)).cpu().squeeze().numpy()
        save_plot((img.permute(1,2,0).numpy()*255).astype(np.uint8),
                  (msk.squeeze().numpy()*255).astype(np.uint8),
                  pr, os.path.join(config.CHECKPOINT_DIR,f"train_vis_{i}.png"))

    # ── Test ──
    test = run_epoch(model, te_dl, criterion)
    with open(os.path.join(config.TEST_OUT_DIR,"test_metrics.txt"),"w",encoding="utf-8") as f:
        for k,v in test.items(): f.write(f"{k}:{v:.6f}\n")
    for i in random.sample(range(len(te_ds)),5):
        img,msk = te_ds[i]
        with torch.no_grad():
            pr = model(img.unsqueeze(0).to(config.DEVICE)).cpu().squeeze().numpy()
        save_plot((img.permute(1,2,0).numpy()*255).astype(np.uint8),
                  (msk.squeeze().numpy()*255).astype(np.uint8),
                  pr, os.path.join(config.TEST_OUT_DIR,f"test_vis_{i}.png"))
    print("\n Métricas Test:", test)

if __name__ == "__main__":
    main()