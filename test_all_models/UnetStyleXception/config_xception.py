import os
import torch


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

# ───────────── Hyperparameters ─────────────
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 6
NUM_EPOCHS    = 150
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 16
NUM_CLASSES   = 1            

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
