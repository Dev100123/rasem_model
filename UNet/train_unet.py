"""
U-Net
TRAIN Model 
Author: Arturo Duarte Rangel 

"""
import os, random, time, cv2, numpy as np, torch, pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from albumentations import (Compose, HorizontalFlip, VerticalFlip,
                            Rotate, RandomBrightnessContrast, Resize, Normalize)
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import config as config

THRESH_VIS = 0.6
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ───────── Dataset ─────────
class BinarySegDataset(Dataset):
    def __init__(self, img_dir, msk_dir, augment=False):
        self.imgs = sorted([os.path.join(img_dir,f) for f in os.listdir(img_dir)
                            if f.lower().endswith((".png",".jpg",".jpeg"))])
        self.msks = sorted([os.path.join(msk_dir,f) for f in os.listdir(msk_dir)
                            if f.lower().endswith((".png",".jpg",".jpeg"))])
        self.aug = augment
        self.t_train = Compose([
            HorizontalFlip(p=0.5), VerticalFlip(p=0.5),
            Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            RandomBrightnessContrast(0.2,0.2,p=0.3),
            Resize(*config.IMAGE_SIZE),
            Normalize(mean=config.MEAN, std=config.STD),
            ToTensorV2(transpose_mask=True)
        ])
        self.t_plain = Compose([
            Resize(*config.IMAGE_SIZE),
            Normalize(mean=config.MEAN, std=config.STD),
            ToTensorV2(transpose_mask=True)
        ])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.msks[idx], cv2.IMREAD_GRAYSCALE)
        data = (self.t_train if self.aug else self.t_plain)(image=img, mask=mask)
        x = data["image"].float()
        y = data["mask"].unsqueeze(0).float()/255.
        return x, y, img_path

# ───────── Modelo U-Net ─────────
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, num_classes=1, base_c=64):
        super().__init__()
        self.enc1 = DoubleConv(3, base_c)
        self.enc2 = DoubleConv(base_c, base_c*2)
        self.enc3 = DoubleConv(base_c*2, base_c*4)
        self.enc4 = DoubleConv(base_c*4, base_c*8)
        self.pool = nn.MaxPool2d(2,2)
        self.bottleneck = DoubleConv(base_c*8, base_c*16)
        self.up4 = nn.ConvTranspose2d(base_c*16, base_c*8, 2, stride=2)
        self.dec4 = DoubleConv(base_c*16, base_c*8)
        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, 2, stride=2)
        self.dec3 = DoubleConv(base_c*8, base_c*4)
        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, 2, stride=2)
        self.dec2 = DoubleConv(base_c*4, base_c*2)
        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, 2, stride=2)
        self.dec1 = DoubleConv(base_c*2, base_c)
        self.outc = nn.Conv2d(base_c, num_classes, 1)
    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.outc(d1)

def get_model(): return UNet(num_classes=config.NUM_CLASSES)

# ───────── Pérdida y métricas ─────────
def dice_coeff(logits, target, smooth=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob*target).sum(dim=(2,3))
    union = prob.sum(dim=(2,3))+target.sum(dim=(2,3))
    return ((2*inter+smooth)/(union+smooth)).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=config.BCE_POS_WEIGHT):
        super().__init__()
        self.register_buffer("pos_w", torch.tensor([pos_weight]))
    def forward(self,p,t):
        bce = nn.functional.binary_cross_entropy_with_logits(
            p, t, pos_weight=self.pos_w.to(p.device))
        return bce + (1-dice_coeff(p,t))

@torch.no_grad()
def batch_metrics(log, tgt, t=0.5, eps=1e-7):
    prob = torch.sigmoid(log)
    pred = (prob>t).float()
    acc  = (pred==tgt).float().mean().item()
    rmse = torch.sqrt(((prob-tgt)**2).mean()).item()
    inter=(pred*tgt).sum(); union=pred.sum()+tgt.sum()-inter+eps
    return acc, rmse, (inter/union).item()

# ───────── Utilities ─────────
def dict_to_df(summary: dict) -> pd.DataFrame:
    filas=[]
    for k in ("loss","acc","rmse","iou","t_img"):
        filas.append({"metrica":k,
                      "media":   summary[k],
                      "desv_std":summary[k+"_sd"]})
    return pd.DataFrame(filas)

# ───────── Epochs ─────────
def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train(train)
    logs = {k:[] for k in ("loss","acc","rmse","iou","t_img")}
    for x,y,_ in (p:=tqdm(loader,leave=False,desc="train" if train else "eval")):
        x,y = x.to(config.DEVICE), y.to(config.DEVICE)
        tic = time.perf_counter()
        with torch.set_grad_enabled(train):
            log = model(x)
            loss = crit(log,y)
            if train:
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if config.DEVICE=="cuda": torch.cuda.synchronize()
        dt = time.perf_counter()-tic
        acc, rmse, iou = batch_metrics(log.detach(),y)
        logs["loss"].append(loss.item())
        logs["acc" ].append(acc)
        logs["rmse"].append(rmse)
        logs["iou" ].append(iou)
        logs["t_img"].append(dt/x.size(0))
        p.set_description(f"{'T' if train else 'V'} L:{loss.item():.3f} IoU:{iou:.3f}")
    summary={}
    for k,v in logs.items():
        summary[k]      = float(np.mean(v))
        summary[k+"_sd"]= float(np.std(v,ddof=0))
    return summary

# ───────── Visualization ─────────
def save_panel(img_path, mask_tgt, logit, path):
    img_vis = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_vis = cv2.resize(img_vis, config.IMAGE_SIZE)
    gt  = (mask_tgt.squeeze().cpu().numpy()>0.5).astype(np.uint8)*255
    pr  = (torch.sigmoid(logit).cpu().squeeze().numpy()>THRESH_VIS).astype(np.uint8)*255
    overlay = img_vis.copy(); overlay[pr==255]=(255,0,0)
    fig,ax = plt.subplots(1,4,figsize=(12,3))
    for a,t,i in zip(ax,["Original","GT","Pred","Overlay"],
                     [img_vis,gt,pr,overlay]):
        a.imshow(i, cmap=None if t=="Original" else "gray")
        a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig(path,dpi=200); plt.close()

# ───────── Main ─────────
def main():
    tr_ds = BinarySegDataset(config.TRAIN_IMAGES_DIR,config.TRAIN_MASKS_DIR,True)
    vl_ds = BinarySegDataset(config.VAL_IMAGES_DIR,config.VAL_MASKS_DIR,False)
    te_ds = BinarySegDataset(config.TEST_IMAGES_DIR,config.TEST_MASKS_DIR,False)
    tr_dl = DataLoader(tr_ds,batch_size=config.BATCH_SIZE,shuffle=True,
                       num_workers=config.NUM_WORKERS,pin_memory=True)
    vl_dl = DataLoader(vl_ds,batch_size=config.BATCH_SIZE,shuffle=False,
                       num_workers=config.NUM_WORKERS,pin_memory=True)
    te_dl = DataLoader(te_ds,batch_size=config.BATCH_SIZE,shuffle=False,
                       num_workers=config.NUM_WORKERS,pin_memory=True)

    model = get_model().to(config.DEVICE)
    crit  = BCEDiceLoss()
    opt   = optim.AdamW(model.parameters(),lr=config.LEARNING_RATE,
                        weight_decay=config.WEIGHT_DECAY)
    sch   = optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=5)

    history,best = [],0.0
    for ep in range(1,config.NUM_EPOCHS+1):
        train_sum = run_epoch(model,tr_dl,crit,opt)
        val_sum   = run_epoch(model,vl_dl,crit)
        history.append({"epoch":ep,"train":train_sum,"val":val_sum})
        if val_sum["iou"]>best:
            best=val_sum["iou"]
            torch.save(model.state_dict(),
                       os.path.join(config.CHECKPOINT_DIR,"unet_best.pth"))
            dict_to_df(val_sum).to_csv(
                os.path.join(config.CHECKPOINT_DIR,"val_metrics_best.csv"),
                index=False)
        sch.step(val_sum["iou"])
        if ep%10==0:
            print(f"Ep {ep}/{config.NUM_EPOCHS} | "
                  f"IoU train {train_sum['iou']:.3f}±{train_sum['iou_sd']:.3f} | "
                  f"IoU val {val_sum['iou']:.3f}±{val_sum['iou_sd']:.3f}")

    # ── save history ──
    rows=[]
    for h in history:
        ep=h["epoch"]
        for split in ("train","val"):
            for m in ("loss","acc","rmse","iou","t_img"):
                rows.append({"epoch":ep,"split":split,
                             "metrica":m,
                             "media":h[split][m],
                             "desv_std":h[split][m+"_sd"]})
    pd.DataFrame(rows).to_csv(
        os.path.join(config.CHECKPOINT_DIR,"train_metrics.csv"), index=False)

    # ── show ──
    model.eval()
    for i in random.sample(range(len(tr_ds)),5):
        x,y,img_path = tr_ds[i]
        with torch.no_grad():
            log = model(x.unsqueeze(0).to(config.DEVICE)).cpu().squeeze(0)
        save_panel(img_path,y,log,
                   os.path.join(config.CHECKPOINT_DIR,f"train_vis_{i}.png"))

    # ── TEST ──
    model.load_state_dict(torch.load(
        os.path.join(config.CHECKPOINT_DIR,"unet_best.pth"),
        map_location=config.DEVICE))
    test_sum = run_epoch(model,te_dl,crit)
    dict_to_df(test_sum).to_csv(
        os.path.join(config.TEST_OUT_DIR,"test_metrics.csv"), index=False)
    for i in random.sample(range(len(te_ds)),5):
        x,y,img_path = te_ds[i]
        with torch.no_grad():
            log = model(x.unsqueeze(0).to(config.DEVICE)).cpu().squeeze(0)
        save_panel(img_path,y,log,
                   os.path.join(config.TEST_OUT_DIR,f"test_vis_{i}.png"))

if __name__=="__main__":
    main()