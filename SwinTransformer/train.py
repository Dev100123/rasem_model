"""
TRAIN Model 
Author: Arturo Duarte Rangel 
"""


import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import NopalDataset
from model import SwinUNet
from config import TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, VAL_IMAGES_DIR, VAL_MASKS_DIR, BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS, LEARNING_RATE, IMAGE_SIZE
import torchvision.transforms as transforms

class RandomFlipRotate:
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        # Flip horizontal
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        # Flip vertical
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        # Rotación en múltiplos de 90 grados
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        return {"image": image, "mask": mask}

class ToTensor:
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        image = image.astype('float32') / 255.0
        image = image.transpose(2, 0, 1)  # de HWC a CHW
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)
        return {"image": image, "mask": mask}

transform = transforms.Compose([
    RandomFlipRotate(),
    ToTensor()
])


def dice_loss(pred, target, eps=1e-6):
    pred_softmax = F.softmax(pred, dim=1)
    prob_nopal = pred_softmax[:, 1:2, :, :]
    target_nopal = (target == 1).float().unsqueeze(1)
    intersection = torch.sum(prob_nopal * target_nopal, dim=(1,2,3))
    union = torch.sum(prob_nopal, dim=(1,2,3)) + torch.sum(target_nopal, dim=(1,2,3)) + eps
    dice = 2.0 * intersection / union
    return 1.0 - dice.mean()


def combined_loss(pred, target, ce_loss_fn):
    ce = ce_loss_fn(pred, target)
    d = dice_loss(pred, target)
    return ce + 0.5 * d

def train():
    train_dataset = NopalDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, transform=transform)
    val_dataset = NopalDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SwinUNet(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks, ce_loss_fn)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks, ce_loss_fn)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completada. Loss entrenamiento: {running_loss / len(train_loader):.4f}, Loss validación: {avg_val_loss:.4f}")

        # checkpoint
        os.makedirs("Modelos", exist_ok=True)
        checkpoint_path = os.path.join("Modelos", f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint guardado en {checkpoint_path}")

if __name__ == '__main__':
    train()