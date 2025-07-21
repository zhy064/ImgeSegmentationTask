import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='val', augment=False):
        self.root = os.path.join(root_dir, 'VOCdevkit', 'VOC2012')
        self.images_dir = os.path.join(self.root, 'JPEGImages')
        self.masks_dir = os.path.join(self.root, 'SegmentationClass')
        self.augment = augment


        self.base_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.base_mask_transform = transforms.ToTensor()


        if self.augment:
            self.image_augment_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        

        split_file = os.path.join(self.root, 'ImageSets', 'Segmentation', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
        self.num_classes = 21

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, f'{img_name}.jpg')
        mask_path = os.path.join(self.masks_dir, f'{img_name}.png')

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        image = image.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)

        if self.augment:
            if torch.rand(1) < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            image = self.image_augment_transform(image)
            
        image_tensor = self.base_image_transform(image)
        mask_tensor = self.base_mask_transform(mask) * 255.0
        mask_tensor = mask_tensor.to(torch.long)

        return image_tensor, mask_tensor.squeeze(0)


def mixup_data(x, y, alpha=1.0, device='cpu'):
    """
    对一个批次的数据应用 Mixup 增强。
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    indices = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[indices, :]
    y_a, y_b = y, y[indices]
    
    return mixed_x, y_a, y_b, lam

def denormalize(tensor):
    """反归一化图像Tensor，方便可视化"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    tensor = (tensor * std + mean).clip(0, 1)
    return tensor

def get_voc_colormap():
    return np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
        [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
        [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ], dtype=np.uint8)

def decode_segmap(mask):
    mask = mask.cpu().numpy()
    colormap = get_voc_colormap()
    r, g, b = np.zeros_like(mask, dtype=np.uint8), np.zeros_like(mask, dtype=np.uint8), np.zeros_like(mask, dtype=np.uint8)
    for class_idx in range(len(colormap)):
        idx = mask == class_idx
        r[idx], g[idx], b[idx] = colormap[class_idx]
    rgb_mask = np.stack([r, g, b], axis=-1)
    return rgb_mask


if __name__ == '__main__':
    ROOT_DIR = "./data"
    voc_path = os.path.join(ROOT_DIR, 'VOCdevkit', 'VOC2012')
    BATCH_SIZE = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")

    #基础图像增强（翻转、颜色抖动等）
    dataset_no_aug = VOCSegmentationDataset(root_dir=ROOT_DIR, split='val', augment=False)
    dataset_with_aug = VOCSegmentationDataset(root_dir=ROOT_DIR, split='val', augment=True)

    sample_idx = 10
    img_no_aug, mask_no_aug = dataset_no_aug[sample_idx]
    img_with_aug, mask_with_aug = dataset_with_aug[sample_idx]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('基础数据增强效果对比 (翻转+颜色抖动)', fontsize=16)

    axes[0, 0].imshow(denormalize(img_no_aug))
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(decode_segmap(mask_no_aug))
    axes[0, 1].set_title('原始掩码')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(denormalize(img_with_aug))
    axes[1, 0].set_title('增强后图像')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(decode_segmap(mask_with_aug))
    axes[1, 1].set_title('增强后掩码')
    axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    #Mixup 数据增强
    dataloader_with_aug = DataLoader(dataset_with_aug, batch_size=BATCH_SIZE, shuffle=True)
    
    images, masks = next(iter(dataloader_with_aug))
    images, masks = images.to(DEVICE), masks.to(DEVICE)

    mixed_images, _, _, lam = mixup_data(images, masks, alpha=0.4, device=DEVICE)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Mixup 增强效果 (λ = {lam:.2f})', fontsize=16)

    axes[0].imshow(denormalize(images[0]))
    axes[0].set_title('原始图像 A (Image A)')
    axes[0].axis('off')

    axes[1].imshow(denormalize(images[1]))
    axes[1].set_title('原始图像 B (Image B)')
    axes[1].axis('off')

    axes[2].imshow(denormalize(mixed_images[0]))
    axes[2].set_title(f'混合图像 (Mixed Image)\n{lam:.2f} * A + {1-lam:.2f} * B')
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()