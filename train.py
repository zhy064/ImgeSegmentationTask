import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

from voc_dataset import VOCSegmentationDataset
from metrics import compute_confusion_matrix, compute_metrics_from_confusion_matrix

def plot_metrics(history, save_path):
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    # plt.title('Training and Validation Loss')
    plt.title('Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], 'gs-', label='Validation Accuracy')
    plt.plot(epochs, history['val_miou'], 'ms-', label='Validation mIoU')
    plt.plot(epochs, history['val_dice'], 'ys-', label='Validation Dice')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"指标曲线图已保存到 {save_path}")
    plt.show()

def main():
    DATA_DIR = 'data'
    OUTPUT_DIR = 'output'
    NUM_EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    NUM_CLASSES = 21
    MODEL_SAVE_PATH = 'best_model_finetuned.pth'
    PLOT_SAVE_PATH = os.path.join(OUTPUT_DIR, 'training_curves_finetuned.png')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = VOCSegmentationDataset(root_dir=DATA_DIR, split='train')
    val_dataset = VOCSegmentationDataset(root_dir=DATA_DIR, split='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.to(device)


    for param in model.backbone.parameters():
        param.requires_grad = False


    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_miou = 0.0
    history = {
        'train_loss': [], 'val_loss': [], 
        'val_acc': [], 'val_miou': [], 'val_dice': []
    }

    for epoch in range(NUM_EPOCHS):
        model.train() 
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({'loss': loss.item()})
        epoch_train_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        total_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                true_masks = masks.cpu().numpy()
                for i in range(images.size(0)):
                    total_conf_matrix += compute_confusion_matrix(preds[i], true_masks[i], NUM_CLASSES)
        
        val_acc, val_miou, val_dice = compute_metrics_from_confusion_matrix(total_conf_matrix)
        epoch_val_loss = val_loss / len(val_dataset)

        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_acc)
        history['val_miou'].append(val_miou)
        history['val_dice'].append(val_dice)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val mIoU: {val_miou:.4f} | Val Dice: {val_dice:.4f}")
        # scheduler.step()
        print(f"  Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved to {MODEL_SAVE_PATH} (mIoU: {best_miou:.4f})")
    
    print("\n训练完成！")
    print(f"最优 mIoU: {best_miou:.4f}")

    plot_metrics(history, PLOT_SAVE_PATH)

if __name__ == '__main__':
    main()