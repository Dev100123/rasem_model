"""
FINE TUNNING 

Unet Expansion — Nopal → Maguey → Nopal+Maguey
Phases
0) Pre-test: Direct inference on Maguey dataset.
1) Fine-Tuning: Freeze backbone layers 0-3 and train for 30 epochs on Maguey dataset.
2) Consolidation: Unfreeze all layers and train for the Mixed (Nopal+Maguey) dataset.
"""

import os, random, time, cv2, numpy as np, torch, pandas as pd
from copy import deepcopy
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import config as config
from train_unet import UNet
# ───────────── Seeds ─────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ───────────── Dataset ─────────────
class SimpleDataset(Dataset):
    
    def __init__(self, img_dir, msk_dir):
        self.imgs = sorted([os.path.join(img_dir,f) for f in os.listdir(img_dir)
                            if f.lower().endswith((".png",".jpg",".jpeg"))])
        self.msks = sorted([os.path.join(msk_dir,f) for f in os.listdir(msk_dir)
                            if f.lower().endswith((".png",".jpg",".jpeg"))])
        self.t = Compose([
            Resize(*config.IMAGE_SIZE),
            Normalize(mean=config.MEAN, std=config.STD),
            ToTensorV2(transpose_mask=True)
        ])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.msks[idx], cv2.IMREAD_GRAYSCALE)
        data = self.t(image=img, mask=msk)
        x = data["image"].float()
        y = data["mask"].unsqueeze(0).float() / 255.
        img_raw = cv2.resize(img, config.IMAGE_SIZE[::-1])
        return x, y, torch.from_numpy(img_raw)

# ───────────── Loss and metrics ─────────────
THR = 0.35
def dice_coeff(logits, target, smooth=1.0):
    prob = torch.sigmoid(logits)
    inter=(prob*target).sum(dim=(2,3))
    union=prob.sum(dim=(2,3))+target.sum(dim=(2,3))
    return ((2*inter+smooth)/(union+smooth)).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=config.BCE_POS_WEIGHT):
        super().__init__()
        self.register_buffer("w", torch.tensor([pos_weight]))
    def forward(self,p,t):
        bce  = nn.functional.binary_cross_entropy_with_logits(
            p,t,pos_weight=self.w.to(p.device))
        return bce + (1-dice_coeff(p,t))

@torch.no_grad()
def batch_metrics(log, tgt):
    prob = torch.sigmoid(log)
    pred = (prob>THR).float()
    acc  = (pred==tgt).float().mean().item()
    rmse = torch.sqrt(((prob-tgt)**2).mean()).item()
    inter=(pred*tgt).sum(); union=pred.sum()+tgt.sum()-inter+1e-7
    iou  = (inter/union).item()
    return acc, rmse, iou

def dict_stats(logs):
    out={}
    for k,v in logs.items():
        out[k]      = float(np.mean(v))
        out[k+"_sd"]= float(np.std(v,ddof=0))
    return out

# ───────────── Epochs ─────────────
def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train(train)
    logs = {k:[] for k in ("loss","acc","rmse","iou","t_img")}
    for x,y,_ in (p:=tqdm(loader,leave=False)):
        x,y = x.to(config.DEVICE), y.to(config.DEVICE)
        tic=time.perf_counter()
        with torch.set_grad_enabled(train):
            log = model(x); loss = crit(log,y)
            if train:
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if config.DEVICE=="cuda": torch.cuda.synchronize()
        dt=time.perf_counter()-tic
        acc,rmse,iou = batch_metrics(log.detach(),y)
        logs["loss"].append(loss.item())
        logs["acc" ].append(acc)
        logs["rmse"].append(rmse)
        logs["iou" ].append(iou)
        logs["t_img"].append(dt/x.size(0))
        p.set_description(f"{'T' if train else 'V'} L:{loss.item():.3f} IoU:{iou:.3f}")
    return dict_stats(logs)

# ───────────── Utility ─────────────
def save_df(path, summary):
    pd.DataFrame([summary]).to_csv(path, index=False)

@torch.no_grad()
def save_examples(model, dataset, out_dir, prefix, n=3):
    os.makedirs(out_dir, exist_ok=True)
    for i in random.sample(range(len(dataset)), n):
        x,y,img_raw = dataset[i]
        log = model(x.unsqueeze(0).to(config.DEVICE)).cpu().squeeze()
        img = img_raw.numpy().astype(np.uint8)
        gt  = (y.squeeze().numpy()>0.5).astype(np.uint8)*255
        pr  = (torch.sigmoid(log).numpy()>THR).astype(np.uint8)*255
        ov  = img.copy(); ov[pr==255]=(255,0,0)
        fig,ax=plt.subplots(1,4,figsize=(12,3))
        for a,t,im in zip(ax,["Orig","GT","Pred","Overlay"],[img,gt,pr,ov]):
            a.imshow(im,cmap=None if t=="Orig" else "gray"); a.set_title(t); a.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"{prefix}_{i}.png"),dpi=200); plt.close()

# ───────────── Main ─────────────
def main():
    # Path
    DIR_PRE  = os.path.join(config.FT_DIR_UNET, "Pretest")
    DIR_FT   = os.path.join(config.FT_DIR_UNET, "FineTune")
    DIR_CONS = os.path.join(config.FT_DIR_UNET, "Consolidation")
    for d in (DIR_PRE, DIR_FT, DIR_CONS): os.makedirs(d, exist_ok=True)

    # Datasets
    n_tr = SimpleDataset(config.TRAIN_IMAGES_DIR , config.TRAIN_MASKS_DIR )
    n_va = SimpleDataset(config.VAL_IMAGES_DIR  , config.VAL_MASKS_DIR  )
    n_te = SimpleDataset(config.TEST_IMAGES_DIR , config.TEST_MASKS_DIR )

    m_tr = SimpleDataset(config.TRAIN_IMAGES_DIRM, config.TRAIN_MASKS_DIRM)
    m_va = SimpleDataset(config.VAL_IMAGES_DIRM  , config.VAL_MASKS_DIRM)
    m_te = SimpleDataset(config.TEST_IMAGES_DIRM , config.TEST_MASKS_DIRM)

    def make_dl(ds, shuffle=False):
        return DataLoader(ds,batch_size=config.BATCH_SIZE,shuffle=shuffle,
                          num_workers=config.NUM_WORKERS,pin_memory=True)
    dl = {
        "n_tr":make_dl(n_tr,True), "n_va":make_dl(n_va), "n_te":make_dl(n_te),
        "m_tr":make_dl(m_tr,True), "m_va":make_dl(m_va), "m_te":make_dl(m_te),
        "mix_tr":make_dl(ConcatDataset([n_tr,m_tr]),True),
        "mix_va":make_dl(ConcatDataset([n_va,m_va]))
    }

    # Model (nopal)
    model = UNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    ckpt  = os.path.join(config.CHECKPOINT_DIR,"unet_best.pth")
    assert os.path.exists(ckpt), f"Checkpoint no encontrado: {ckpt}"
    model.load_state_dict(torch.load(ckpt,map_location=config.DEVICE))
    crit = BCEDiceLoss()

    # ═══ 0. Pre-test maguey ═══
    print("\n Pre-test (inferencia directa en maguey)")
    pre = run_epoch(model, dl["m_te"], crit)
    save_df(os.path.join(DIR_PRE,"metrics_pretest.csv"), pre)
    save_examples(model,m_te,DIR_PRE,"pre",n=3)

    # ═══ 1. Fine-Tuning ═══
    print("\n Fine-Tuning congelando codificador")
    for name,p in model.named_parameters():
        if name.startswith(("enc1","enc2","enc3","enc4")):
            p.requires_grad = False
    opt_ft = optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),
                         lr=1e-4,weight_decay=1e-5)
    best_iou, BEST_W = (

        .0, None); EPOCHS_FT=30
    for ep in range(1,EPOCHS_FT+1):
        tr = run_epoch(model, dl["m_tr"], crit, opt_ft)
        vl = run_epoch(model, dl["m_va"], crit)
        print(f"FT {ep:02d}/{EPOCHS_FT} | IoU Train {tr['iou']:.3f} | IoU Val {vl['iou']:.3f}")
        if vl["iou"]>best_iou:
            best_iou, BEST_W = vl["iou"], deepcopy(model.state_dict())
    torch.save(BEST_W, os.path.join(DIR_FT,"unet_finetuned.pth"))
    model.load_state_dict(BEST_W)

    ft_test = run_epoch(model, dl["m_te"], crit)
    save_df(os.path.join(DIR_FT,"metrics_finetune.csv"), ft_test)
    save_examples(model,m_te,DIR_FT,"ft",n=5)

    # ═══ 2. Consolidation ═══
    print("\n🔄 Consolidación nopal + maguey")
    for p in model.parameters(): p.requires_grad=True
    opt_c = optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-5)
    EPOCHS_C=10
    for ep in range(1,EPOCHS_C+1):
        run_epoch(model, dl["mix_tr"], crit, opt_c)
        vl = run_epoch(model, dl["mix_va"], crit)
        print(f"CONS {ep}/{EPOCHS_C} | IoU Mix Val {vl['iou']:.3f}")
    torch.save(model.state_dict(), os.path.join(DIR_CONS,"unet_final.pth"))

    # Metrics
    final_m = run_epoch(model, dl["m_te"], crit)
    final_n = run_epoch(model, dl["n_te"], crit)
    save_df(os.path.join(DIR_CONS,"metrics_maguey.csv"), final_m)
    save_df(os.path.join(DIR_CONS,"metrics_nopal.csv"),  final_n)
    save_examples(model,m_te,DIR_CONS,"mag_fin",n=3)
    save_examples(model,n_tr,DIR_CONS,"nop_fin",n=3)

    print("\n Proceso completed → unet_final.pth")

if __name__ == "__main__":
    main()