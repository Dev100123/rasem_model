# -*- coding: utf-8 -*-
"""
SegNet with PyTorch
Author: Arturo Duarte Rangel 
"""
import os, random, json, time
import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import config

# ───────────────── Dataset ─────────────────
class NopalDataset(Dataset):
    def __init__(self, img_dir, msk_dir):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
        self.msk_paths = sorted([os.path.join(msk_dir, f) for f in os.listdir(msk_dir) if f.endswith(".png")])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img  = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        msk  = cv2.imread(self.msk_paths[idx], cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img, config.IMAGE_SIZE[::-1]) / 255.0
        msk  = cv2.resize(msk, config.IMAGE_SIZE[::-1]) / 255.0
        msk  = (msk > 0.5).astype(np.float32)
        img  = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)
        msk  = torch.tensor(msk, dtype=torch.float32).unsqueeze(0)
        return img, msk

# ───────────── SegNet ─────────────
def conv_bn_relu(in_c, out_c, k=3):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=k//2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))

class SegNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.enc1 = nn.Sequential(conv_bn_relu(3, 64), conv_bn_relu(64, 64))
        self.enc2 = nn.Sequential(conv_bn_relu(64,128), conv_bn_relu(128,128))
        self.enc3 = nn.Sequential(conv_bn_relu(128,256), conv_bn_relu(256,256), conv_bn_relu(256,256))
        self.enc4 = nn.Sequential(conv_bn_relu(256,512), conv_bn_relu(512,512), conv_bn_relu(512,512))
        self.enc5 = nn.Sequential(conv_bn_relu(512,512), conv_bn_relu(512,512), conv_bn_relu(512,512))
        self.pool   = nn.MaxPool2d(2,2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2,2)
        self.dec5 = nn.Sequential(conv_bn_relu(512,512), conv_bn_relu(512,512), conv_bn_relu(512,512))
        self.dec4 = nn.Sequential(conv_bn_relu(512,512), conv_bn_relu(512,512), conv_bn_relu(512,256))
        self.dec3 = nn.Sequential(conv_bn_relu(256,256), conv_bn_relu(256,256), conv_bn_relu(256,128))
        self.dec2 = nn.Sequential(conv_bn_relu(128,128), conv_bn_relu(128,64))
        self.dec1 = nn.Sequential(conv_bn_relu(64,64), nn.Conv2d(64,n_classes,1))
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

# ───────────── Metrics ─────────────
def metrics(pred, mask):
    pred_bin = (pred>0.5).float()
    acc  = (pred_bin==mask).float().mean().item()
    rmse = torch.sqrt(((pred-mask)**2).mean()).item()
    inter = (pred_bin*mask).sum()
    union = pred_bin.sum()+mask.sum()-inter +1e-7
    iou = (inter/union).item()
    return acc, rmse, iou

# ───────────── Epochs ─────────────
def run_epoch(model, loader, criterion, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    agg = {"loss":0,"acc":0,"rmse":0,"iou":0}
    t_tot, n_img = 0.0, 0
    for imgs, msks in (pbar:=tqdm(loader, leave=False)):
        imgs, msks = imgs.to(config.DEVICE), msks.to(config.DEVICE)
        start = time.perf_counter()
        with torch.set_grad_enabled(train):
            outs = model(imgs)
            loss = criterion(outs, msks)
            if train:
                optim.zero_grad(); loss.backward(); optim.step()
        torch.cuda.synchronize() if config.DEVICE=="cuda" else None
        t_tot += time.perf_counter()-start
        n_img += imgs.size(0)

        acc, rmse, iou = metrics(outs.detach(), msks)
        agg["loss"] += loss.item(); agg["acc"]+=acc; agg["rmse"]+=rmse; agg["iou"]+=iou
        pbar.set_description(f"L:{loss.item():.3f} IoU:{iou:.3f}")
    n=len(loader)
    agg = {k:v/n for k,v in agg.items()}
    agg["t_img"] = t_tot/n_img
    return agg

# ───────────── Visual ─────────────
def save_plot(img, msk, pred, path):
    pred_bin = (pred>0.5).astype(np.uint8)*255
    overlay = img.copy(); overlay[pred_bin==255]=(255,0,0)
    fig,ax = plt.subplots(1,4,figsize=(12,3))
    ax[0].imshow(img);            ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(msk,cmap="gray");ax[1].set_title("GT");       ax[1].axis("off")
    ax[2].imshow(pred_bin,cmap="gray"); ax[2].set_title("Pred"); ax[2].axis("off")
    ax[3].imshow(overlay);        ax[3].set_title("Overlay");  ax[3].axis("off")
    plt.tight_layout(); plt.savefig(path); plt.close()

# ───────────── Main ─────────────
def main():
    # Data
    tr_ds = NopalDataset(config.TRAIN_IMAGES_DIR, config.TRAIN_MASKS_DIR)
    vl_ds = NopalDataset(config.VAL_IMAGES_DIR,   config.VAL_MASKS_DIR)
    te_ds = NopalDataset(config.TEST_IMAGES_DIR,  config.TEST_MASKS_DIR)
    tr_dl = DataLoader(tr_ds,batch_size=config.BATCH_SIZE,shuffle=True,
                       num_workers=config.NUM_WORKERS,pin_memory=True)
    vl_dl = DataLoader(vl_ds,batch_size=config.BATCH_SIZE,shuffle=False,
                       num_workers=config.NUM_WORKERS,pin_memory=True)
    te_dl = DataLoader(te_ds,batch_size=config.BATCH_SIZE,shuffle=False,
                       num_workers=config.NUM_WORKERS,pin_memory=True)

    # Model
    model = SegNet(config.NUM_CLASSES).to(config.DEVICE)
    crit = nn.BCELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                              weight_decay=config.WEIGHT_DECAY)

    history, best_iou = [], 0
    print(f"\n  Dispositivo: {config.DEVICE.upper()}  |  Épocas: {config.NUM_EPOCHS}")
    for epoch in range(1, config.NUM_EPOCHS+1):
        tr = run_epoch(model, tr_dl, crit, optim)
        vl = run_epoch(model, vl_dl, crit)
        history.append({"epoch":epoch,"train":tr,"val":vl})
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
              f"Train L:{tr['loss']:.4f} Acc:{tr['acc']:.4f} RMSE:{tr['rmse']:.4f} IoU:{tr['iou']:.4f} t:{tr['t_img']*1e3:.2f}ms || "
              f"Val L:{vl['loss']:.4f} Acc:{vl['acc']:.4f} RMSE:{vl['rmse']:.4f} IoU:{vl['iou']:.4f} t:{vl['t_img']*1e3:.2f}ms")
        if vl["iou"]>best_iou:
            best_iou=vl["iou"]
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR,"segnet_best.pth"))

    # ──────
    def write_metrics_txt(path, hist):
        keys = ["loss","acc","rmse","iou","t_img"]
        
        with open(path,"w",encoding="utf-8") as f:
            f.write("# epoch " + " ".join([f"train_{k}" for k in keys]) +
                    " " + " ".join([f"val_{k}" for k in keys]) + "\n")
            # por época
            for h in hist:
                vals = [h["train"][k] for k in keys] + [h["val"][k] for k in keys]
                f.write(f"{h['epoch']:3d} " + " ".join([f"{v:.6f}" for v in vals]) + "\n")
            # resumen
            def arr(split,k): return np.array([h[split][k] for h in hist])
            f.write("\n# media ± std\n")
            for split in ["train","val"]:
                f.write(f"{split.capitalize():5s}: " + " | ".join(
                    [f"{k}:{arr(split,k).mean():.4f}±{arr(split,k).std():.4f}" for k in keys]) + "\n")
    write_metrics_txt(os.path.join(config.CHECKPOINT_DIR,"train_metrics.txt"), history)

    # ─── TRAIN (5) ───
    for i in random.sample(range(len(tr_ds)),5):
        img,msk = tr_ds[i]
        with torch.no_grad():
            pr = model(img.unsqueeze(0).to(config.DEVICE)).cpu().squeeze().numpy()
        save_plot((img.permute(1,2,0).numpy()*255).astype(np.uint8),
                  (msk.squeeze().numpy()*255).astype(np.uint8),
                  pr, os.path.join(config.CHECKPOINT_DIR,f"train_vis_{i}.png"))

    # ─── Test ───
    test = run_epoch(model, te_dl, crit)
    print("\n Test:", test)

    # guarda métricas test
    with open(os.path.join(config.TEST_OUT_DIR,"test_metrics.txt"),"w",encoding="utf-8") as f:
        for k,v in test.items():
            f.write(f"{k}:{v:.6f}\n")

    # test
    for i in random.sample(range(len(te_ds)),5):
        img,msk = te_ds[i]
        with torch.no_grad():
            pr = model(img.unsqueeze(0).to(config.DEVICE)).cpu().squeeze().numpy()
        save_plot((img.permute(1,2,0).numpy()*255).astype(np.uint8),
                  (msk.squeeze().numpy()*255).astype(np.uint8),
                  pr, os.path.join(config.TEST_OUT_DIR,f"test_vis_{i}.png"))

if __name__ == "__main__":
    main()