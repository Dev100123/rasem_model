import os

# ───────────────── dataset (Opuntia spp.) ─────────────────
DATA_DIR = os.path.join(os.getcwd(), "../Seg_Net/Dataset_Nopal")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "Train", "Images_Aumentada")
TRAIN_MASKS_DIR  = os.path.join(DATA_DIR, "Train", "Masks_binaria_Aumentada")
VAL_IMAGES_DIR   = os.path.join(DATA_DIR, "Validacion", "Images_Aumentada")
VAL_MASKS_DIR    = os.path.join(DATA_DIR, "Validacion", "Masks_binaria_Aumentada")
TEST_IMAGES_DIR  = os.path.join(DATA_DIR, "Test", "Images_Aumentada")
TEST_MASKS_DIR   = os.path.join(DATA_DIR, "Test", "Masks_binaria_Aumentada")

# ──────────────── maguey ─────────────
DATA_DIRM        = os.path.join(os.getcwd(), "../Seg_Net/Dataset_Maguey")
TRAIN_IMAGES_DIRM = os.path.join(DATA_DIRM, "train", "imagenes")
TRAIN_MASKS_DIRM = os.path.join(DATA_DIRM, "train", "mascaras")
VAL_IMAGES_DIRM   = os.path.join(DATA_DIRM, "val", "imagenes")
VAL_MASKS_DIRM    = os.path.join(DATA_DIRM, "val", "mascaras")
TEST_IMAGES_DIRM = os.path.join(DATA_DIRM, "test", "imagenes")
TEST_MASKS_DIRM = os.path.join(DATA_DIRM, "test", "mascaras")

# ───────────────── Config ─────────────────
BATCH_SIZE    = 4
NUM_WORKERS   = 2
NUM_CLASSES   = 1
IMAGE_SIZE    = (224, 224)
LEARNING_RATE = 1e-4
NUM_EPOCHS    = 150

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

OUT_DIR = os.path.join(os.getcwd(), "Modelos")
os.makedirs(OUT_DIR, exist_ok=True)