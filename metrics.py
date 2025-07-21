
import numpy as np

def compute_confusion_matrix(pred_mask, true_mask, num_classes):
    # 忽略标签值为 255 的像素
    mask = (true_mask >= 0) & (true_mask < num_classes)
    hist = np.bincount(
        num_classes * true_mask[mask].astype(int) + pred_mask[mask],
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return hist

def compute_metrics_from_confusion_matrix(conf_matrix):
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp
    pixel_accuracy = tp.sum() / (conf_matrix.sum() + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    mean_iou = np.nanmean(iou) 
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    mean_dice = np.nanmean(dice)
    return pixel_accuracy, mean_iou, mean_dice