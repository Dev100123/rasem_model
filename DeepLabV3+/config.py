"""
Global configuration
"""
import os
import torch

# ────────────────────dataset ────────────────────
DATA_DIR = os.path.join(os.getcwd(), "../Seg_Net/Dataset_Nopal")

PROJECT_DIR = os.path.join(os.getcwd(), "")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "Train",      "Images_Aumentada")
TRAIN_MASKS_DIR  = os.path.join(DATA_DIR, "Train",      "Masks_binaria_Aumentada")

VAL_IMAGES_DIR   = os.path.join(DATA_DIR, "Validacion", "Images_Aumentada")
VAL_MASKS_DIR    = os.path.join(DATA_DIR, "Validacion", "Masks_binaria_Aumentada")

TEST_IMAGES_DIR  = os.path.join(DATA_DIR, "Test",       "Images_Aumentada")
TEST_MASKS_DIR   = os.path.join(DATA_DIR, "Test",       "Masks_binaria_Aumentada")


DATA_DIRM         = os.path.join(os.getcwd(), "../Seg_Net/Dataset_Maguey")
TRAIN_IMAGES_DIRM = os.path.join(DATA_DIRM, "train", "imagenes")
TRAIN_MASKS_DIRM  = os.path.join(DATA_DIRM, "train", "mascaras")
VAL_IMAGES_DIRM   = os.path.join(DATA_DIRM, "val", "imagenes")
VAL_MASKS_DIRM    = os.path.join(DATA_DIRM, "val", "mascaras")
TEST_IMAGES_DIRM  = os.path.join(DATA_DIRM, "test", "imagenes")
TEST_MASKS_DIRM   = os.path.join(DATA_DIRM, "test", "mascaras")

# ══════════════════════════════
IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = 1
BATCH_SIZE    = 8
NUM_EPOCHS    = 150
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
NUM_WORKERS   = 8
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


BCE_POS_WEIGHT = 3.0
DICE_SMOOTH    = 1.0

# ═══════════════ 
CHECKPOINT_DIR = os.path.join(os.getcwd(), "Modelo")
TEST_OUT_DIR   = os.path.join(os.getcwd(), "Test_Nopal")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TEST_OUT_DIR,   exist_ok=True)
# ═══════════════ 
FT_DIR_DL = os.path.join(os.getcwd(), "FT_DIR_DL")
os.makedirs(FT_DIR_DL, exist_ok=True)
# ═══════════════ 
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

