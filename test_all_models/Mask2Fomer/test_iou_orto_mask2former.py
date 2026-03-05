import os, glob, csv, time, argparse
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Mask2FormerForUniversalSegmentation

from config_mask2former import IMAGE_SIZE, MEAN, STD

cultivo = "maguey"#"nopal"  # o
IMAGES_DIR = r"../output_images_" + cultivo
MASKS_DIR  = r"../output_masks_"  + cultivo
WEIGHTS    = r"../Modelos_Consolidacion_best/modelo_mask2former_best_5.pth"
OUT_DIR    = r"Test_Mask2Former_Consolidado_" + cultivo
IMG_EXTS   = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

def _stem(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem

def _find_mask_for_image(img_path: str, masks_dir: str):
    stem = _stem(img_path)
    for ext in IMG_EXTS:
        cand = os.path.join(masks_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    candidates = glob.glob(os.path.join(masks_dir, stem + ".*"))
    return candidates[0] if candidates else None

# ───────── Dataset de pares ─────────
class PairDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, image_size, mean, std):
        
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.image_size = image_size  # (H, W)
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)

        files = [f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)]
        self.img_files = sorted([os.path.join(images_dir, f) for f in files])
        if not self.img_files:
            raise RuntimeError(f"No se encontraron imágenes en: {images_dir}")

        self.pairs = []
        for ip in self.img_files:
            mp = _find_mask_for_image(ip, masks_dir)
            if mp is None:
                raise RuntimeError(f"No se encontró máscara para:\n  {ip}\n en: {masks_dir}")
            self.pairs.append((ip, mp))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]

        # overlay
        img_bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"No se pudo leer imagen: {ip}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_rgb = img_rgb.copy()  # (H0, W0, 3) uint8

        # MASK
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise RuntimeError(f"No se pudo leer máscara: {mp}")
        if msk.max() > 1:
            msk = (msk > 127).astype(np.uint8)

        # MODEL
        H, W = self.image_size
        img_res = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img_res = (img_res - self.mean) / self.std
        img_res = img_res.transpose(2, 0, 1)  # C,H,W

        msk_res = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        return {
            "name": os.path.basename(ip),
            "img_tensor": torch.from_numpy(img_res).float(),  # (3,H,W)
            "mask_resized": torch.from_numpy(msk_res).byte(), # (H,W) {0,1}
            "orig_rgb": orig_rgb,                             # (H0,W0,3) uint8
        }

# ──────────────────
def iou_numpy(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or (pred_bin, gt_bin).sum() + 1e-6
    return float(inter) / float(union)

def make_red_overlay(orig_rgb: np.ndarray, pred_bin: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay = orig_rgb.copy()
    red = np.zeros_like(orig_rgb); red[..., 0] = 255  # canal R
    m = pred_bin.astype(bool)
    overlay[m] = (alpha * red[m] + (1 - alpha) * overlay[m]).astype(np.uint8)
    return overlay

# ──────────────────
@torch.no_grad()
def compute_binary_logit_from_outputs(outputs):
    
    class_logits = getattr(outputs, "pred_logits", None)
    masks_logits = getattr(outputs, "pred_masks",  None)
    if class_logits is None:
        class_logits = getattr(outputs, "class_queries_logits", None)
    if masks_logits is None:
        masks_logits = getattr(outputs, "masks_queries_logits",  None)

    if masks_logits is None:
        raise RuntimeError("No se encontraron tensores de máscaras en las salidas del modelo.")

    if class_logits is None:
        # Fallback
        return masks_logits.max(dim=1).values  # [B,h,w]

    # class_logits: [B,Q,C]; masks_logits: [B,Q,h,w]
    B, Q, H, W = masks_logits.shape
    C = class_logits.shape[-1]
    cls_prob = torch.softmax(class_logits, dim=-1)  # [B,Q,C]

    if C == 1:
        weights = torch.ones((B, Q, 1), device=masks_logits.device, dtype=masks_logits.dtype)
    elif C == 2:
        # Foreground → índice 1 como foreground
        weights = cls_prob[:, :, 1:2]
    else:
        # Excluir último índice típico 'no-object'
        weights = cls_prob[:, :, :-1].max(dim=-1, keepdim=True).values  # [B,Q,1]

    weights = weights.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # [B,1,Q,1,1]
    masks   = masks_logits.unsqueeze(1)                             # [B,1,Q,H,W]
    combined = (weights * masks).sum(dim=2).squeeze(1)              # [B,H,W]
    return combined

# ───────── test ─────────
@torch.no_grad()
def run_test(images_dir: str, masks_dir: str, weights_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_over = os.path.join(out_dir, "overlays")
    os.makedirs(out_over, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-ade-semantic",
        ignore_mismatched_sizes=True
    ).to(device)
    model.eval()

    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    ds = PairDataset(images_dir, masks_dir, IMAGE_SIZE, MEAN, STD)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    ious, times = [], []

    for b in ld:
        name = b["name"][0]
        x = b["img_tensor"].to(device)               # (1,3,H,W)
        gt_r = b["mask_resized"].squeeze(0).numpy().astype(np.uint8)

        orig_arr = b["orig_rgb"]
        if isinstance(orig_arr, list):
            orig = orig_arr[0]
        elif isinstance(orig_arr, torch.Tensor):
            orig = orig_arr[0].cpu().numpy()
        elif isinstance(orig_arr, np.ndarray):
            orig = orig_arr[0] if orig_arr.ndim == 4 else orig_arr
        else:
            orig = np.array(orig_arr)[0]
        H0, W0 = orig.shape[:2]

        t0 = time.perf_counter()
        outputs = model(pixel_values=x)
        bin_logit = compute_binary_logit_from_outputs(outputs)  # [1,h,w]
        if bin_logit.shape[-2:] != gt_r.shape[-2:]:
            bin_logit = torch.nn.functional.interpolate(
                bin_logit.unsqueeze(1), size=gt_r.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)

        prob = torch.sigmoid(bin_logit)[:, :]        # (1,H,W)
        pred_res = (prob > 0.5).cpu().numpy()[0].astype(np.uint8)

        ious.append(iou_numpy(pred_res, gt_r))

        pred_orig = cv2.resize(pred_res, (W0, H0), interpolation=cv2.INTER_NEAREST)
        overlay = make_red_overlay(orig, pred_orig, alpha=0.5)

        out_path = os.path.join(out_over, os.path.splitext(name)[0] + "_overlay.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    iou_mean = float(np.mean(ious)) if ious else float("nan")
    iou_std  = float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0
    t_total  = float(np.sum(times)) if times else float("nan")
    t_mean   = float(np.mean(times)) if times else float("nan")
    t_std    = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0

    # CSV
    csv_path = os.path.join(out_dir, f"metricas_test_mask2former_{cultivo}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "media", "desv_std"])
        w.writerow(["IoU", iou_mean, iou_std])
        w.writerow(["t_total", t_total, 0.0])  # tiempo total del conjunto (s)
        w.writerow(["t_img", t_mean, t_std])   # promedio ± σ por imagen (s)

    print("\n Test completado.")
    print(f"IoU:     {iou_mean:.6f} ± {iou_std:.6f}")
    print(f"t_total: {t_total:.6f}s (conjunto completo)")
    print(f"t_img:   {t_mean:.6f}s ± {t_std:.6f}s (por imagen)")
    print(f"Overlays guardados en: {out_over}")
    print(f"Métricas guardadas en: {csv_path}")

# ───────── CLI ─────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=IMAGES_DIR, help="Carpeta de imágenes RGB")
    ap.add_argument("--masks",  default=MASKS_DIR,  help="Carpeta de máscaras binarias")
    ap.add_argument("--weights",default=WEIGHTS,    help="Ruta a pesos .pth")
    ap.add_argument("--out",    default=OUT_DIR,    help="Carpeta de salida")
    return ap.parse_args()

# ───────── Entry point ─────────
if __name__ == "__main__":
    args = parse_args()
    run_test(args.images, args.masks, args.weights, args.out)