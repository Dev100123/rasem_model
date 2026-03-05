import os
import csv
import time
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

cultivo="maguey" ##"nopal"
# ─────────────────────────── CONFIGURACIÓN DEL USUARIO ───────────────────────────
IMAGES_DIR = r"../output_images_"+cultivo
MASKS_DIR  = r"../output_masks_"+cultivo
WEIGHTS    = r"../Modelos_Consolidacion_best/modelo_segnet_best_3.pth"
OUT_DIR    = r"Test_SegNet_Consolidado_"+cultivo

# ─────────────────────────── IMPORTS ───────────────────────────────

from config_segnet import IMAGE_SIZE, DEVICE        # IMAGE_SIZE = (altura, anchura)
CV2_SIZE = (IMAGE_SIZE[1], IMAGE_SIZE[0])    # (anchura, altura) para cv2

# callback.
try:
    from Seg_Net_Model_Metricas import SegNet  # fine-tuning
except Exception:
    import torch.nn as nn
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
            return torch.sigmoid(d1)  # salida ya en [0,1]

# ─────────────────────────── DATASET ────────────────────────────────────────────
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

def _stem(path):
    b = os.path.basename(path)
    s, _ = os.path.splitext(b)
    return s

def _find_mask_for_image(img_path, masks_dir):

    stem = _stem(img_path)
    for ext in IMG_EXTS:
        cand = os.path.join(masks_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    candidates = glob.glob(os.path.join(masks_dir, stem + ".*"))
    return candidates[0] if candidates else None

class PairDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir

        self.img_files = sorted([
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if f.lower().endswith(IMG_EXTS)
        ])
        if not self.img_files:
            raise RuntimeError(f"No se encontraron imágenes en: {images_dir}")

        self.pairs = []
        for ip in self.img_files:
            mp = _find_mask_for_image(ip, masks_dir)
            if mp is None:
                raise RuntimeError(f"No se encontró máscara para la imagen:\n  {ip}\n"
                                   f"en la carpeta: {masks_dir}")
            self.pairs.append((ip, mp))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]

        img_bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"No se pudo leer la imagen: {ip}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_rgb = img_rgb.copy()

        img_res = cv2.resize(img_rgb, CV2_SIZE, interpolation=cv2.INTER_LINEAR)
        img_res = (img_res.astype(np.float32) / 255.0).transpose(2, 0, 1)  # C,H,W

        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise RuntimeError(f"No se pudo leer la máscara: {mp}")
        if msk.max() > 1:
            msk = (msk > 127).astype(np.uint8)

        msk_res = cv2.resize(msk, CV2_SIZE, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        return {
            "name": os.path.basename(ip),
            "img_tensor": torch.from_numpy(img_res).float(),  # (3,H,W) [0,1]
            "mask_resized": torch.from_numpy(msk_res).byte(), # (H,W) {0,1}
            "orig_rgb": orig_rgb,                             # (H0,W0,3) uint8
        }

# ─────────────────────────── METRICS Y OVERLAY ─────────────────────────────────
def iou_numpy(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or (pred_bin, gt_bin).sum() + 1e-6
    return inter / union

def make_red_overlay(orig_rgb, pred_bin, alpha=0.5):
    overlay = orig_rgb.copy()
    red = np.zeros_like(orig_rgb)
    red[..., 0] = 255  # canal R
    m = pred_bin.astype(bool)
    overlay[m] = (alpha * red[m] + (1 - alpha) * overlay[m]).astype(np.uint8)
    return overlay

# ─────────────────────────── TEST ───────────────────────────────────────
def run_test(images_dir, masks_dir, weights_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_over = os.path.join(out_dir, "overlays")
    os.makedirs(out_over, exist_ok=True)

    # Device
    device = torch.device(DEVICE)
    print("Dispositivo:", device)

    # Model
    model = SegNet(n_classes=1).to(device)
    state = torch.load(weights_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    # Data
    ds = PairDataset(images_dir, masks_dir)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    ious, times = [], []

    with torch.no_grad():
        for b in ld:
            name = b["name"][0]
            x = b["img_tensor"].to(device)            # (1,3,H,W)
            gt_r = b["mask_resized"].squeeze(0).numpy().astype(np.uint8)

            orig_arr = b["orig_rgb"]
            if isinstance(orig_arr, torch.Tensor):
                orig = orig_arr[0].cpu().numpy()
            elif isinstance(orig_arr, np.ndarray):
                orig = orig_arr[0] if orig_arr.ndim == 4 else orig_arr
            else:
                orig = np.array(orig_arr)[0]
            H0, W0 = orig.shape[:2]

            # Inference timing
            t0 = time.perf_counter()
            prob = model(x)                           # sigmoide (1,1,H,W)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)

            pred_res = (prob[:, 0] > 0.5).cpu().numpy()[0].astype(np.uint8)

            # IoU
            ious.append(iou_numpy(pred_res, gt_r))

            # Visual
            pred_orig = cv2.resize(pred_res, (W0, H0), interpolation=cv2.INTER_NEAREST)
            overlay = make_red_overlay(orig, pred_orig, alpha=0.5)

            out_path = os.path.join(out_over, os.path.splitext(name)[0] + "_overlay.png")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    iou_mean = float(np.mean(ious)) if len(ious) else float("nan")
    iou_std  = float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0

    t_total  = float(np.sum(times)) if len(times) else float("nan")
    t_mean   = float(np.mean(times)) if len(times) else float("nan")
    t_std    = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0

    # CSV
    csv_path = os.path.join(out_dir, "metricas_test_segnet_"+cultivo+".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "media", "desv_std"])
        w.writerow(["IoU",     iou_mean, iou_std])
        w.writerow(["t_total", t_total,  0.0])
        w.writerow(["t_img",   t_mean,   t_std])

    print("\n Test completado.")
    print(f"IoU:     {iou_mean:.6f} ± {iou_std:.6f}")
    print(f"t_total: {t_total:.6f}s (conjunto completo)")
    print(f"t_img:   {t_mean:.6f}s ± {t_std:.6f}s (por imagen)")
    print(f"Overlays guardados en: {out_over}")
    print(f"Métricas guardadas en: {csv_path}")

# ─────────────────────────── MAIN ───────────────────────────────
if __name__ == "__main__":
    run_test(IMAGES_DIR, MASKS_DIR, WEIGHTS, OUT_DIR)