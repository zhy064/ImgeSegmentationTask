import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from voc_dataset import VOCSegmentationDataset
from inference import decode_segmap

def main():
    DATA_DIR = 'data'
    OUTPUT_DIR = 'output' 
    FINETUNED_MODEL_PATH = 'best_model_finetuned.pth' 
    NUM_IMAGES_TO_SHOW = 3
    os.makedirs(OUTPUT_DIR, exist_ok=True) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    original_model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    original_model.to(device)
    original_model.eval()


    #加载微调后模型
    finetuned_model = models.segmentation.deeplabv3_resnet101(num_classes=21, aux_loss=True)
    finetuned_model.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=device))
    finetuned_model.to(device)
    finetuned_model.eval()


    val_dataset = VOCSegmentationDataset(root_dir=DATA_DIR, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    data_iter = iter(val_loader)


    for i in range(NUM_IMAGES_TO_SHOW):

        image_tensor, mask_tensor = next(data_iter)
        image_tensor_device = image_tensor.to(device)

        with torch.no_grad():
            out_original = original_model(image_tensor_device)['out']
            pred_original = torch.argmax(out_original, dim=1).squeeze(0).cpu().numpy()

            out_finetuned = finetuned_model(image_tensor_device)['out']
            pred_finetuned = torch.argmax(out_finetuned, dim=1).squeeze(0).cpu().numpy()

        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img_display = to_pil_image(inv_normalize(image_tensor.squeeze(0)))
        
        gt_mask_color = decode_segmap(mask_tensor.squeeze(0).numpy())
        pred_original_color = decode_segmap(pred_original)
        pred_finetuned_color = decode_segmap(pred_finetuned)
        
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        
        axs[0].imshow(img_display); axs[0].set_title('Original Image'); axs[0].axis('off')
        axs[1].imshow(gt_mask_color); axs[1].set_title('Ground Truth Mask'); axs[1].axis('off')
        axs[2].imshow(pred_original_color); axs[2].set_title('Prediction (Fine-Tuned Model)'); axs[2].axis('off')
        axs[3].imshow(pred_finetuned_color); axs[3].set_title('Prediction (Original Model)'); axs[3].axis('off')

        plt.suptitle(f'Comparison Result - Sample {i+1}', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f'comparison_result_{i+1}.png') 
        plt.savefig(save_path)
        plt.show()

if __name__ == '__main__':
    main()