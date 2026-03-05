import os
import csv
import time
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

cultivo="maguey" ##"nopal"
# ─────────────────────────── CONFIG ───────────────────────────
IMAGES_DIR = r"../output_images_"+cultivo
MASKS_DIR  = r"../output_masks_"+cultivo
WEIGHTS    = r"../Modelos_Consolidacion_best/modelo_swintransformer_best_5.pth"                           # ← pesos consolidados (.pth)
OUT_DIR    = r"Test_Swin_Consolidado_"+cultivo

# ─────────────────────────── IMPORTS ───────────────────────────────
from model  import SwinUNet
from config_SwinTransformer import IMAGE_SIZE

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
    def __init__(self, images_dir, masks_dir, image_size):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.image_size = image_size

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

        # overlay
        img_bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"No se pudo leer la imagen: {ip}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_rgb = img_rgb.copy()

        img_res = cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        img_res = (img_res.astype(np.float32) / 255.0).transpose(2, 0, 1)  # C,H,W

        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise RuntimeError(f"No se pudo leer la máscara: {mp}")
        if msk.max() > 1:
            msk = (msk > 127).astype(np.uint8)

        msk_res = cv2.resize(msk, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

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

# ───────────────────────────TEST ───────────────────────────────────────
def run_test(images_dir, masks_dir, weights_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_over = os.path.join(out_dir, "overlays")
    os.makedirs(out_over, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    # Model
    model = SwinUNet(num_classes=1).to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    # Data
    ds = PairDataset(images_dir, masks_dir, IMAGE_SIZE)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    ious, times = [], []

    with torch.no_grad():
        for b in ld:
            name = b["name"][0]

            x = b["img_tensor"].to(device)  # (1,3,H,W)

            gt_r = b["mask_resized"].squeeze(0).numpy().astype(np.uint8)  # (H,W)

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
            ylog = model(x)  # logits (1,1,H,W)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)

            prob = torch.sigmoid(ylog)[:, 0]            # (1,H,W)
            pred_res = (prob > 0.5).cpu().numpy()[0].astype(np.uint8)  # (H,W)

            ious.append(iou_numpy(pred_res, gt_r))

            pred_orig = cv2.resize(pred_res, (W0, H0), interpolation=cv2.INTER_NEAREST)

            # Overlay red
            overlay = make_red_overlay(orig, pred_orig, alpha=0.5)

            # save overlay
            out_path = os.path.join(out_over, os.path.splitext(name)[0] + "_overlay.png")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    iou_mean = float(np.mean(ious)) if len(ious) else float("nan")
    iou_std = float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0

    t_total = float(np.sum(times)) if len(times) else float("nan")
    t_mean = float(np.mean(times)) if len(times) else float("nan")
    t_std = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0

    # CSV
    csv_path = os.path.join(out_dir, "metricas_test_swinTransformer_"+cultivo+".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "media", "desv_std"])
        w.writerow(["IoU", iou_mean, iou_std])
        w.writerow(["t_total", t_total, 0.0])
        w.writerow(["t_img", t_mean, t_std])

    print("\n✓ Test completado.")
    print(f"IoU:     {iou_mean:.6f} ± {iou_std:.6f}")
    print(f"t_total: {t_total:.6f}s (conjunto completo)")
    print(f"t_img:   {t_mean:.6f}s ± {t_std:.6f}s (por imagen)")
    print(f"Overlays guardados en: {out_over}")
    print(f"Métricas guardadas en: {csv_path}")

# ─────────────────────────── Main ───────────────────────────────
if __name__ == "__main__":
    run_test(IMAGES_DIR, MASKS_DIR, WEIGHTS, OUT_DIR)