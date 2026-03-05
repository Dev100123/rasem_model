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
WEIGHTS    = r"../Modelos_Consolidacion_best/modelo_unet_best_5.pth"
OUT_DIR    = r"Test_UNet_Consolidado_"+cultivo

# ─────────────────────────── IMPORTS DEL PROYECTO ───────────────────────────────
#import   #config_unet.py
from config_Unet import IMAGE_SIZE, DEVICE, MEAN, STD
CV2_SIZE = (IMAGE_SIZE[1], IMAGE_SIZE[0])

try:
    from Unet.train_unet import UNet
except Exception:
    try:
        from train_unet import UNet
    except Exception as e:
        raise ImportError(
            "Could not import UNet from 'train_unet' or 'train'. "
            "Adjust the import according to the file where you defined the model."
        ) from e

# ─────────────────────────── HELPERS ────────────────────────────────────────────
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

def as_numpy_uint8(img):
    if isinstance(img, (list, tuple)):
        img = img[0]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if img.ndim == 4:
        img = img[0]
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img

# ─────────────────────────── DATASET ────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir

        self.img_files = sorted([
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if f.lower().endswith(IMG_EXTS)
        ])
        if not self.img_files:
            raise RuntimeError(f"No images were found in: {images_dir}")

        self.pairs = []
        for ip in self.img_files:
            mp = _find_mask_for_image(ip, masks_dir)
            if mp is None:
                raise RuntimeError(f"No images were found in:\n  {ip}\n"
                                   f"in the folder: {masks_dir}")
            self.pairs.append((ip, mp))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]

        # Image (BGR→RGB) and an original copy for overlay
        img_bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Could not read the image: {ip}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_rgb = img_rgb.copy()

        # Resize and normalize (ImageNet MEAN/STD as in training)
        img_res = cv2.resize(img_rgb, CV2_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = img_res.transpose(2, 0, 1)  # C,H,W
        mean = np.array(MEAN, dtype=np.float32)[:, None, None]
        std  = np.array(STD,  dtype=np.float32)[:, None, None]
        x = (x - mean) / std

        # Mask (grayscale → binary {0,1}) and resize
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise RuntimeError(f"Could not read the mask: {mp}")
        if msk.max() > 1:
            msk = (msk > 127).astype(np.uint8)
        msk_res = cv2.resize(msk, CV2_SIZE, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        return {
            "name": os.path.basename(ip),
            "img_tensor": torch.from_numpy(x).float(),
            "mask_resized": torch.from_numpy(msk_res).byte(),
            "orig_rgb": orig_rgb,
        }

# ─────────────────────────── MÉTRIC and OVERLAY ─────────────────────────────────
def iou_numpy(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or (pred_bin, gt_bin).sum() + 1e-6
    return inter / union

def make_red_overlay(orig_rgb, pred_bin, alpha=0.5):
    orig_rgb = as_numpy_uint8(orig_rgb)
    overlay = orig_rgb.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    m = pred_bin.astype(bool)
    overlay[m] = (alpha * red[m] + (1 - alpha) * overlay[m]).astype(np.uint8)
    return overlay

# ─────────────────────────── LOOP DE TEST ───────────────────────────────────────
def run_test(images_dir, masks_dir, weights_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_over = os.path.join(out_dir, "overlays")
    os.makedirs(out_over, exist_ok=True)

    # Device:
    device = torch.device(DEVICE)
    print("Device:", device)

    # Model
    model = UNet(num_classes=1).to(device)  # salida: logits
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
            x    = b["img_tensor"].to(device)
            gt_r = b["mask_resized"].squeeze(0).numpy().astype(np.uint8)

            # Original image (for overlay)
            orig = as_numpy_uint8(b["orig_rgb"])
            H0, W0 = orig.shape[:2]

            # Inference timing
            t0 = time.perf_counter()
            logits = model(x.unsqueeze(0) if x.ndim == 3 else x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)

            # Sigmoid → binary
            prob = torch.sigmoid(logits)[:, 0]
            pred_res = (prob > 0.5).cpu().numpy()[0].astype(np.uint8)

            # IoU at the model input size
            ious.append(iou_numpy(pred_res, gt_r))

            # Visualization at the original size
            pred_orig = cv2.resize(pred_res, (W0, H0), interpolation=cv2.INTER_NEAREST)
            overlay = make_red_overlay(orig, pred_orig, alpha=0.5)

            out_path = os.path.join(out_over, os.path.splitext(name)[0] + "_overlay.png")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Statistics
    iou_mean = float(np.mean(ious)) if len(ious) else float("nan")
    iou_std  = float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0

    t_total  = float(np.sum(times)) if len(times) else float("nan")
    t_mean   = float(np.mean(times)) if len(times) else float("nan")
    t_std    = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0

    # CSV
    csv_path = os.path.join(out_dir, "metricas_test_unet_"+cultivo+".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "desv_std"])
        w.writerow(["IoU",     iou_mean, iou_std])
        w.writerow(["t_total", t_total,  0.0])
        w.writerow(["t_img",   t_mean,   t_std])

    print("\n✓ Test completed.")
    print(f"IoU:     {iou_mean:.6f} ± {iou_std:.6f}")
    print(f"t_total: {t_total:.6f}s (Full dataset)")
    print(f"t_img:   {t_mean:.6f}s ± {t_std:.6f}s (per image)")
    print(f"Overlays saved to: {out_over}")
    print(f"Metrics saved to: {csv_path}")

# ─────────────────────────── RUNNING ───────────────────────────────
if __name__ == "__main__":
    run_test(IMAGES_DIR, MASKS_DIR, WEIGHTS, OUT_DIR)