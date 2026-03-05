"""
train Swin-UNet
"""
import os, time, csv, random, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dataset import NopalDataset
from model   import SwinUNet
from config  import (TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR,
                     VAL_IMAGES_DIR,   VAL_MASKS_DIR,
                     BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS, LEARNING_RATE)
import torchvision.transforms as transforms

# ──────────────────────────────────
class RandomFlipRotate:
    def __call__(self, s):
        img, msk = s["image"], s["mask"]
        if random.random() > .5: img, msk = np.fliplr(img).copy(), np.fliplr(msk).copy()
        if random.random() > .5: img, msk = np.flipud(img).copy(), np.flipud(msk).copy()
        k = random.randint(0,3)
        if k: img, msk = np.rot90(img,k).copy(), np.rot90(msk,k).copy()
        return {"image": img, "mask": msk}

class ToTensor:
    def __call__(self, s):
        img, msk = s["image"], s["mask"]
        img = (img.astype(np.float32)/255.).transpose(2,0,1)
        return {"image": torch.tensor(img), "mask": torch.tensor(msk, dtype=torch.long)}

transform = transforms.Compose([RandomFlipRotate(), ToTensor()])

# ──────────────────────────────────────────
def compute_metrics(logits, masks):
    prob  = torch.sigmoid(logits)[:,0]      # B,H,W
    pred  = (prob > .5).float()
    maskf = masks.float()
    acc  = (pred == maskf).float().mean()
    rmse = torch.sqrt(((prob-maskf)**2).mean())
    inter = (pred*maskf).sum()
    union = pred.sum()+maskf.sum()-inter+1e-6
    iou  = inter/union
    return acc.item(), rmse.item(), iou.item()

def dice_loss(p,t,eps=1e-6):
    p=torch.sigmoid(p); t=t.float().unsqueeze(1)
    inter=(p*t).sum((1,2,3)); union=p.sum((1,2,3))+t.sum((1,2,3))+eps
    return 1-(2*inter/union).mean()

def combined_loss(p,t,bce): return bce(p,t.float().unsqueeze(1))+0.5*dice_loss(p,t)

# ───────────────── Main ────────────────────
def main():

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Dispositivo:', device)
    model = SwinUNet(num_classes=1).to(device)
    opt   = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce   = nn.BCEWithLogitsLoss()

    tr_ds=NopalDataset(TRAIN_IMAGES_DIR,TRAIN_MASKS_DIR,transform)
    va_ds=NopalDataset(VAL_IMAGES_DIR,  VAL_MASKS_DIR,  transform)
    tr_ld=DataLoader(tr_ds,batch_size=BATCH_SIZE,shuffle=True, num_workers=NUM_WORKERS)
    va_ld=DataLoader(va_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)

    hist={k:[] for k in ["train_loss","train_acc","train_rmse","train_iou",
                         "val_loss","val_acc","val_rmse","val_iou","t_img"]}

    best_iou=-1.0
    os.makedirs("Modelos",exist_ok=True)

    for ep in range(1, NUM_EPOCHS+1):
        # ---------- Train ----------
        model.train(); tl=ta=tr=ti=0
        for b in tr_ld:
            img,msk=b["image"].to(device),b["mask"].to(device)
            opt.zero_grad(); out=model(img)
            loss=combined_loss(out,msk,bce); loss.backward(); opt.step()
            l,a,r,i=loss.item(),*compute_metrics(out,msk)
            tl+=l; ta+=a; tr+=r; ti+=i
        ntr=len(tr_ld); tl,ta,tr,ti=[x/ntr for x in (tl,ta,tr,ti)]

        # ---------- Val ----------
        model.eval(); vl=va=vr=vi=t_sum=n_img=0
        with torch.no_grad():
            for b in va_ld:
                img,msk=b["image"].to(device),b["mask"].to(device)
                t0=time.perf_counter(); out=model(img)
                t_sum+=time.perf_counter()-t0; n_img+=img.size(0)
                l,a,r,i=combined_loss(out,msk,bce).item(),*compute_metrics(out,msk)
                vl+=l; va+=a; vr+=r; vi+=i
        nva=len(va_ld); vl,va,vr,vi=[x/nva for x in (vl,va,vr,vi)]
        t_img=t_sum/n_img

        # Guarda mejor modelo
        if vi>best_iou:
            best_iou=vi
            torch.save(model.state_dict(),
                       os.path.join("Modelos",f"modelo_best.pth"))

        # Historial
        for k,v in zip(hist.keys(),
                       [tl,ta,tr,ti,vl,va,vr,vi,t_img]):
            hist[k].append(v)

        if ep%50==0:# or ep==1:
            print(f"Ep {ep:03d}/{NUM_EPOCHS} | "
                  f"Train L:{tl:.4f} Acc:{ta:.4f} RMSE:{tr:.4f} IoU:{ti:.4f} || "
                  f"Val L:{vl:.4f} Acc:{va:.4f} RMSE:{vr:.4f} IoU:{vi:.4f} | t_img:{t_img:.5f}s")

    # --------- Model ---------
    final_path=os.path.join("Modelos",f"modelo_final150.pth")
    torch.save(model.state_dict(), final_path)
    print("✓ Modelo final guardado en", final_path)

    # --------- CSV resumen ---------
    with open(os.path.join("Modelos",f"metricas_resumen.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["metrica","media","desv_std"])
        for k in ["train_loss","train_acc","train_rmse","train_iou",
                  "val_loss","val_acc","val_rmse","val_iou","t_img"]:
            w.writerow([k, np.mean(hist[k]), np.std(hist[k],ddof=1)])

    print("✓ Resumen de métricas guardado.")

if __name__=="__main__":
    main()