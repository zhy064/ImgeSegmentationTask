import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np
import os 
from voc_dataset import VOCSegmentationDataset 

# 类别颜色映射
VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
], dtype=np.uint8)

def decode_segmap(image, nc=21):
    label_colors = VOC_COLORMAP
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    ignore_idx = image == 255
    r[ignore_idx] = 255
    g[ignore_idx] = 255
    b[ignore_idx] = 255
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def main():
    DATA_DIR = 'data'
    OUTPUT_DIR = 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True) # <-- 如果文件夹不存在，则创建它
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.to(device)
    model.eval() 


    val_dataset = VOCSegmentationDataset(root_dir=DATA_DIR, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


    num_images_to_show = 3
    data_iter = iter(val_loader)

    for i in range(num_images_to_show):
        image_tensor, mask_tensor = next(data_iter)
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            output = model(image_tensor)['out']
            pred_mask = torch.argmax(output, dim=1)

        image_tensor = image_tensor.squeeze(0).cpu()
        pred_mask_np = pred_mask.squeeze(0).cpu().numpy()
        gt_mask_np = mask_tensor.squeeze(0).cpu().numpy()
        
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img_display = to_pil_image(inv_normalize(image_tensor))

        pred_mask_color = decode_segmap(pred_mask_np)
        gt_mask_color = decode_segmap(gt_mask_np)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        axs[0].imshow(img_display)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(gt_mask_color)
        axs[1].set_title('Ground Truth Mask')
        axs[1].axis('off')
        
        axs[2].imshow(pred_mask_color)
        axs[2].set_title('Predicted Mask')
        axs[2].axis('off')

        plt.suptitle(f'Inference Result - Sample {i+1}', fontsize=16)
        
        save_path = os.path.join(OUTPUT_DIR, f'inference_result_{i+1}.png') # <-- 使用 os.path.join 构造路径
        plt.savefig(save_path)
        plt.show()

if __name__ == '__main__':
    main()