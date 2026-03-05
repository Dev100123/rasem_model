import os
import glob
import csv
import time
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────
IMAGES_DIR   = r"../output_images_nopal"
MASKS_DIR    = r"../output_masks_nopal"
WEIGHTS      = r"../Modelos_Consolidacion_best/modelo_swintransformer_best_5.pth"
OUT_DIR      = r"Viz_Swin_Real_20x6"
MAX_SAMPLES  = 20
THRESH       = 0.5
FIG_DPI      = 170
TITLE_FIG    = "Inferencia Real"

# ─────────────────────────── IMPORTS ───────────────────────────────
from model  import SwinUNet
from config_SwinTransformer import IMAGE_SIZE  # (W, H), p. ej. (224, 224)

# ─────────────────────────── UTILITIES ───────────────────────────
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

def _stem(path):
    b = os.path.basename(path); s, _ = os.path.splitext(b); return s

def _find_mask_for_image(img_path, masks_dir):
    stem = _stem(img_path)
    for ext in IMG_EXTS:
        cand = os.path.join(masks_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    cands = glob.glob(os.path.join(masks_dir, stem + ".*"))
    return cands[0] if cands else None

def binarize_mask(msk):
    if msk is None: return None
    if msk.dtype != np.uint8: msk = msk.astype(np.uint8)
    if msk.max() > 1: msk = (msk > 127).astype(np.uint8)
    return msk

def resize_pair_to_model(img_rgb, msk_bin, size_wh):
    w, h = size_wh
    img_res = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    msk_res = cv2.resize(msk_bin, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return img_res, msk_res

def to_tensor(img_rgb_res):
    arr = (img_rgb_res.astype(np.float32) / 255.0).transpose(2, 0, 1)  # C,H,W
    return torch.from_numpy(arr).float()

def iou_numpy(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or (pred_bin, gt_bin).sum() + 1e-6
    return float(inter / union)

def overlay_color(rgb_uint8, mask_bin, color="red", alpha=0.5):

    overlay = rgb_uint8.copy()
    col = np.zeros_like(rgb_uint8)
    if color == "red":
        col[..., 0] = 255
    elif color == "blue":
        col[..., 2] = 255
    m = mask_bin.astype(bool)
    overlay[m] = (alpha * col[m] + (1 - alpha) * overlay[m]).astype(np.uint8)
    return overlay

# ─────────────────────────── DATASET ───────────────────────────
class PairDataset(Dataset):
    def __init__(self, images_dir, masks_dir, max_samples=20):
        img_files = sorted([
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if f.lower().endswith(IMG_EXTS)
        ])
        pairs = []
        for ip in img_files:
            mp = _find_mask_for_image(ip, masks_dir)
            if mp is not None:
                pairs.append((ip, mp))
        if not pairs:
            raise RuntimeError("No se encontraron pares imagen–máscara válidos.")
        self.pairs = pairs[:max_samples]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        # Image RGB
        img_bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"No se pudo leer la imagen: {ip}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # MASK
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise RuntimeError(f"No se pudo leer la máscara: {mp}")
        msk = binarize_mask(msk)
        return {"name": os.path.basename(ip), "img_rgb": img_rgb, "msk_bin": msk}

# ─────────────────────────── INFERENCE ───────────────────────────
def predict_mask(model, device, img_rgb_resized, thresh=0.5):
    """Recibe imagen YA redimensionada a IMAGE_SIZE. Devuelve (mask_bin, dt_segundos)."""
    x = to_tensor(img_rgb_resized)[None, ...].to(device)
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    prob = torch.sigmoid(logits)[:, 0]   # (1,H,W)
    pred = (prob > thresh).float().cpu().numpy()[0].astype(np.uint8)
    return pred, dt

# ─────────────────────────── MAIN ───────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNet(num_classes=1).to(device)
    state = torch.load(WEIGHTS, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    # Dataset
    ds = PairDataset(IMAGES_DIR, MASKS_DIR, max_samples=MAX_SAMPLES)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    n = len(ds)
    fig_h = max(12, int(n * 1.5))
    fig, axes = plt.subplots(nrows=n, ncols=6, figsize=(22, fig_h))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    fig.suptitle(TITLE_FIG, fontsize=16, y=0.995)

    # CSV
    csv_path = os.path.join(OUT_DIR, "metricas_inferencia_real_20x6.csv")
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["nombre",
                    "IoU_original", "IoU_rotada90",
                    "IoU_combinacion_OR",
                    "t_img_orig_s", "t_img_rot_s"])

        for i, batch in enumerate(ld):
            name = batch["name"][0]
            img0  = batch["img_rgb"][0].numpy() if isinstance(batch["img_rgb"], torch.Tensor) else batch["img_rgb"][0]
            gt0   = batch["msk_bin"][0].numpy() if isinstance(batch["msk_bin"], torch.Tensor) else batch["msk_bin"][0]
            if not isinstance(img0, np.ndarray): img0 = np.array(img0)
            if not isinstance(gt0,  np.ndarray): gt0  = np.array(gt0)

            # MODEL
            w_res, h_res = IMAGE_SIZE
            img0_res, gt0_res = resize_pair_to_model(img0, gt0, IMAGE_SIZE)

            # (1) Prediction
            pred_orig, dt_orig = predict_mask(model, device, img0_res, THRESH)
            iou_orig = iou_numpy(pred_orig, gt0_res)
            overlay_orig = overlay_color(img0_res, pred_orig, color="red", alpha=0.5)

            # (2) Prediction
            img_rot_res = cv2.rotate(img0_res, cv2.ROTATE_90_CLOCKWISE)
            gt_rot_res  = cv2.rotate(gt0_res,  cv2.ROTATE_90_CLOCKWISE)
            pred_rot, dt_rot = predict_mask(model, device, img_rot_res, THRESH)
            iou_rot = iou_numpy(pred_rot, gt_rot_res)
            overlay_rot = overlay_color(img_rot_res, pred_rot, color="blue", alpha=0.5)

            # (3) Prediction
            pred_rot_back = cv2.rotate(pred_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)  # (H,W) binaria en pos. original

            # (4)
            blend_base = img0_res.copy()

            blend_red = overlay_color(blend_base, pred_orig, color="red", alpha=0.5)

            blend_both = overlay_color(blend_red, pred_rot_back, color="blue", alpha=0.5)

            # (5)
            combined_or = np.clip(pred_orig | pred_rot_back, 0, 1).astype(np.uint8)
            iou_comb_or = iou_numpy(combined_or, gt0_res)

            # ── row i ──
            ax1, ax2, ax3, ax4, ax5, ax6 = axes[i]

            ax1.imshow(img0_res);         ax1.set_title("Original");            ax1.axis("off")
            ax2.imshow(overlay_orig);     ax2.set_title(f"Pred. Orig (IoU={iou_orig:.4f})"); ax2.axis("off")
            ax3.imshow(img_rot_res);      ax3.set_title("Rotada 90°");          ax3.axis("off")
            ax4.imshow(overlay_rot);      ax4.set_title(f"Pred. Rot (IoU={iou_rot:.4f})");   ax4.axis("off")
            ax5.imshow(blend_both);       ax5.set_title("Solape (rojo+azul)");  ax5.axis("off")

            ax6.axis("off")
            ax6.text(0.05, 0.6,
                     f"IoU combinación (OR)\nvs GT original:\n{iou_comb_or:.4f}",
                     fontsize=11, va="top", ha="left",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

            # CSV
            w.writerow([name,
                        f"{iou_orig:.6f}", f"{iou_rot:.6f}",
                        f"{iou_comb_or:.6f}",
                        f"{dt_orig:.6f}",  f"{dt_rot:.6f}"])

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = os.path.join(OUT_DIR, "inferencias_20x6.png")
    fig.savefig(fig_path, dpi=FIG_DPI)
    plt.close(fig)

    print(" Visualización 20×6 completada.")
    print(f"Figura:   {fig_path}")
    print(f"Métricas: {csv_path}")

# ─────────────────────────── main ───────────────────────────
if __name__ == "__main__":
    main()