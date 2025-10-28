import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv
from segm.metrics import gather_data
import segm.utils.torch as ptu

IGNORE_LABEL = 255

# ----------------------------
# LOSS TRACKING
# ----------------------------
LOSS_HISTORY = {
    "CE": [],
    "Weighted_CE": [],
    "Dice": [],
    "Validation": [],
    "Total": [],
}

# ----------------------------
# LOSS FUNCTIONS
# ----------------------------
def dice_loss(pred, target, smooth=1e-6):
    """Compute Dice loss for binary segmentation (multi-class adaptation included)."""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def plot_losses(log_dir):
    """Plot all tracked losses in one figure (CE, Weighted CE, Dice, Validation, Total)."""
    plt.figure(figsize=(10, 6))
    max_len = max(len(v) for v in LOSS_HISTORY.values())
    x_axis = np.arange(1, max_len + 1)

    # distinct color and marker per loss
    style = {
        "CE": ("blue", "o"),
        "Weighted_CE": ("orange", "s"),
        "Dice": ("green", "D"),
        "Validation": ("red", "x"),
        "Total": ("purple", "^"),
    }

    for key in ["CE", "Weighted_CE", "Dice", "Validation", "Total"]:
        values = LOSS_HISTORY.get(key, [])
        if len(values) > 0:
            plot_values = values + [np.nan] * (max_len - len(values))
            color, marker = style.get(key, ("black", "o"))
            plt.plot(
                x_axis,
                plot_values,
                label=key.replace("_", " "),
                linewidth=2.0,
                color=color,
                marker=marker,
                markersize=6,
                alpha=0.85
            )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Losses", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_losses.png"), dpi=150)
    plt.close()



# ----------------------------
# TRAINING
# ----------------------------
def train_one_epoch(model, data_loader, optimizer, lr_scheduler, epoch, amp_autocast,
                    loss_scaler=None, log_dir=None, class_weights=None, val_loader=None):
    """
    Train model for one epoch and track all meaningful losses.
    """
    model.train()
    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    if class_weights is not None:
        weighted_ce_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(ptu.device), ignore_index=IGNORE_LABEL)
    else:
        weighted_ce_fn = ce_loss_fn

    ce_epoch, weighted_ce_epoch, dice_epoch, total_epoch = 0.0, 0.0, 0.0, 0.0

    for batch in data_loader:
        images = batch["image"].to(ptu.device)
        masks = batch["mask"].to(ptu.device).long()

        optimizer.zero_grad()
        with amp_autocast():
            outputs = model(images)
            ce_loss = ce_loss_fn(outputs, masks)
            weighted_ce_loss = weighted_ce_fn(outputs, masks)
            probs = torch.softmax(outputs, dim=1)

            # Dice
            if probs.shape[1] > 1:
                dice = torch.mean(torch.stack([
                    dice_loss(probs[:, c, :, :], (masks == c).float()) for c in range(probs.shape[1])
                ]))
            else:
                dice = dice_loss(probs[:, 0, :, :], masks.float())

            total_loss = ce_loss + dice

        if loss_scaler is not None:
            loss_scaler(total_loss, optimizer)
        else:
            total_loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        ce_epoch += ce_loss.item()
        weighted_ce_epoch += weighted_ce_loss.item()
        dice_epoch += dice.item()
        total_epoch += total_loss.item()

    n_batches = len(data_loader)
    ce_epoch /= n_batches
    weighted_ce_epoch /= n_batches
    dice_epoch /= n_batches
    total_epoch /= n_batches

    # --- Compute Validation Loss ---
    val_loss_epoch = None
    if val_loader is not None:
        val_loss_epoch = compute_validation_loss(model, val_loader, ce_loss_fn, weighted_ce_fn, amp_autocast)

    # --- Append losses ---
    LOSS_HISTORY["CE"].append(ce_epoch)
    LOSS_HISTORY["Weighted_CE"].append(weighted_ce_epoch)
    LOSS_HISTORY["Dice"].append(dice_epoch)
    LOSS_HISTORY["Validation"].append(val_loss_epoch if val_loss_epoch is not None else np.nan)
    LOSS_HISTORY["Total"].append(total_epoch)

    # Plot and save
    if log_dir:
        plot_losses(log_dir)

    return {
        "CE": ce_epoch,
        "Weighted_CE": weighted_ce_epoch,
        "Dice": dice_epoch,
        "Validation": val_loss_epoch,
        "Total": total_epoch
    }


# (everything below remains 100% unchanged)
# compute_validation_loss, evaluate, compute_segmentation_metrics stay exactly as before
def compute_validation_loss(model, val_loader, ce_fn, weighted_ce_fn, amp_autocast):
    """Compute validation loss for the val_loader."""
    model.eval()
    ce_sum, weighted_ce_sum, dice_sum, total_sum = 0.0, 0.0, 0.0, 0.0
    n_batches = len(val_loader)
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(ptu.device)
            masks = batch["mask"].to(ptu.device).long()
            with amp_autocast():
                outputs = model(images)
                ce_loss = ce_fn(outputs, masks)
                weighted_ce_loss = weighted_ce_fn(outputs, masks)
                probs = torch.softmax(outputs, dim=1)

                if probs.shape[1] > 1:
                    dice = torch.mean(torch.stack([
                        dice_loss(probs[:, c, :, :], (masks == c).float()) for c in range(probs.shape[1])
                    ]))
                else:
                    dice = dice_loss(probs[:, 0, :, :], masks.float())

                total_loss = ce_loss + dice

            ce_sum += ce_loss.item()
            weighted_ce_sum += weighted_ce_loss.item()
            dice_sum += dice.item()
            total_sum += total_loss.item()

    return total_sum / n_batches


# ----------------------------
# VALIDATION & METRICS
# ----------------------------
@torch.no_grad()
def evaluate(model, data_loader, val_seg_gt, window_size=None, window_stride=None, amp_autocast=None, log_dir=None, epoch=None):
    """
    Evaluate model and save full metrics to CSV (PixelAcc, IoU, Dice, etc.)
    """
    model_eval = model.module if hasattr(model, "module") else model
    seg_pred = {}

    for batch in data_loader:
        images = batch["image"].to(ptu.device)
        ids = batch["id"]

        with amp_autocast():
            outputs = model_eval(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        for i, file_id in enumerate(ids):
            pred = preds[i]
            gt = val_seg_gt[file_id]
            if pred.shape != gt.shape:
                import cv2
                pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            seg_pred[file_id] = pred

    seg_pred = gather_data(seg_pred)
    val_seg_gt_filtered = {k: np.asarray(val_seg_gt[k], dtype=np.int64) for k in seg_pred.keys()}

    n_cls = getattr(data_loader.dataset, "n_cls", 2)
    metrics = compute_segmentation_metrics(seg_pred, val_seg_gt_filtered, n_cls)

    # Save metrics to CSV
    if log_dir and epoch is not None:
        csv_path = os.path.join(log_dir, "evaluation_metrics.csv")
        header = ["epoch"] + list(metrics.keys())
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            row = {"epoch": epoch}
            for k, v in metrics.items():
                row[k] = list(v) if isinstance(v, np.ndarray) else v
            writer.writerow(row)

    return metrics


# ----------------------------
# METRICS COMPUTATION
# ----------------------------
def compute_segmentation_metrics(preds, gts, n_cls):
    """
    Compute full segmentation metrics:
    PixelAcc, MeanAcc, IoU, MeanIoU, FWIoU, Dice, Precision, Recall, F1, PerClassIoU
    """
    eps = 1e-6
    total_pixels = 0
    correct_pixels = 0

    dice_per_class = np.zeros(n_cls, dtype=np.float32)
    precision_per_class = np.zeros(n_cls, dtype=np.float32)
    recall_per_class = np.zeros(n_cls, dtype=np.float32)
    iou_per_class = np.zeros(n_cls, dtype=np.float32)
    acc_per_class = np.zeros(n_cls, dtype=np.float32)
    fw_iou_num = 0
    fw_iou_den = 0

    # Count pixels for per-class accuracy
    class_counts = np.zeros(n_cls, dtype=np.float32)

    for k in preds.keys():
        pred = preds[k].flatten()
        gt = gts[k].flatten()
        mask = (gt != IGNORE_LABEL)

        pred = pred[mask]
        gt = gt[mask]

        total_pixels += len(gt)
        correct_pixels += np.sum(pred == gt)

        for c in range(n_cls):
            pred_c = (pred == c)
            gt_c = (gt == c)
            intersection = np.sum(pred_c & gt_c)
            union = np.sum(pred_c | gt_c)
            dice_per_class[c] += (2 * intersection) / (np.sum(pred_c) + np.sum(gt_c) + eps)
            precision_per_class[c] += intersection / (np.sum(pred_c) + eps)
            recall_per_class[c] += intersection / (np.sum(gt_c) + eps)
            iou_per_class[c] += intersection / (union + eps)
            acc_per_class[c] += np.sum(pred_c & gt_c) / (np.sum(gt_c) + eps)
            fw_iou_num += intersection
            fw_iou_den += np.sum(gt_c)
            class_counts[c] += np.sum(gt_c)

    num_images = len(preds)
    per_class_iou = iou_per_class / num_images
    per_class_dice = dice_per_class / num_images

    # Overall Pixel Accuracy
    pixel_acc = correct_pixels / total_pixels
    # Mean per-class Accuracy
    mean_acc = np.mean(acc_per_class / num_images)
    # Mean IoU across classes
    mean_iou = np.mean(per_class_iou)
    # Frequency Weighted IoU (overall IoU)
    fw_iou = fw_iou_num / (fw_iou_den + eps)

    metrics = {
        "PixelAcc": pixel_acc,
        "MeanAcc": mean_acc,
        "IoU": fw_iou,            # overall / weighted IoU
        "MeanIoU": mean_iou,      # average per-class IoU
        "FWIoU": fw_iou,
        "PerClassDice": per_class_dice,
        "Precision": precision_per_class / num_images,
        "Recall": recall_per_class / num_images,
        "F1": 2 * (precision_per_class * recall_per_class) /
              (precision_per_class + recall_per_class + eps),
        "PerClassIoU": per_class_iou
    }

    return metrics



