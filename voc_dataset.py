import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class VOCSegmentationDataset(Dataset):
    """
    PASCAL VOC 2012 语义分割数据集的自定义 Dataset 类。
    """
    def __init__(self, root_dir, split='val', transform=None):
        self.root = os.path.join(root_dir, 'VOCdevkit', 'VOC2012')
        self.images_dir = os.path.join(self.root, 'JPEGImages')
        self.masks_dir = os.path.join(self.root, 'SegmentationClass')
        
        # 1. 定义数据预处理流程
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
            self.mask_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])

        # 2. 读取分割集文件（train.txt 或 val.txt）
        split_file = os.path.join(self.root, 'ImageSets', 'Segmentation', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
        self.num_classes = 21 # PASCAL VOC 有 21 个类别

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, f'{img_name}.jpg')
        mask_path = os.path.join(self.masks_dir, f'{img_name}.png')

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image_tensor = self.transform(image)
        mask_tensor = self.mask_transform(mask) * 255
        mask_tensor = mask_tensor.to(torch.long)

        return image_tensor, mask_tensor.squeeze(0)


# 测试和可视化代码
if __name__ == '__main__':
    DATA_ROOT_DIR = './data' 

    voc_val_dataset = VOCSegmentationDataset(root_dir=DATA_ROOT_DIR, split='val')
    print(f"成功从 '{DATA_ROOT_DIR}' 加载数据集，共计 {len(voc_val_dataset)} 个样本。")
    first_image, first_mask = voc_val_dataset[0]
    print(f"图像张量 (Image Tensor) 的形状: {first_image.shape}") 
    print(f"掩码张量 (Mask Tensor) 的形状:  {first_mask.shape}")
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        tensor = tensor.clone()
        tensor = tensor.permute(1, 2, 0)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor
    def visualize_sample(dataset, index):
        image_tensor, mask_tensor = dataset[index]
        display_image = denormalize(image_tensor)
        mask_numpy = mask_tensor.cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.imshow(display_image)
        ax1.set_title(f'原图 (Image) - Sample #{index}')
        ax1.axis('off') 

        ax2.imshow(mask_numpy, cmap='tab20', vmin=0, vmax=20)
        ax2.set_title(f'分割掩码 (Mask) - Sample #{index}')
        ax2.axis('off') 
        
        plt.tight_layout()
        plt.show()

    visualize_sample(voc_val_dataset, 10)
    visualize_sample(voc_val_dataset, 50)