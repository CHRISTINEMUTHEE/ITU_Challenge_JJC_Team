import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# --- IMPORT FROM CORE MODULES ---
from core.model import build_model
from core.dataset import PixelEmbeddingDataset, PixelFusionDataset, LatentTokenDataset, find_file_pairs, find_pixel_fusion_pairs, HEIGHT_NORM_CONSTANT
from core.losses import ImprovedCompositeLoss

# --- 1. EXPERIMENT TRACKING ---
EXPERIMENT_NAME = "alpha_tessera_fusion_mae/"
BASE_DIR = "./runs"
EXP_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)
VIZ_OUTPUT_DIR = os.path.join(EXP_DIR, "visualizations")

# Paths for saving models and plots
BEST_MODEL_PATH = os.path.join(EXP_DIR, "model_best.pth")
LAST_MODEL_PATH = os.path.join(EXP_DIR, "model_last.pth")
LOSS_CURVE_PATH = os.path.join(EXP_DIR, "loss_curve.png")
CONFIG_LOG_PATH = os.path.join(EXP_DIR, "training_params.txt")

# --- 2. CONFIGURATION ---
# TRAIN_EMBEDDINGS_DIR = "../../emb2heights/data/gee_emb_aligned_v2/"

TRAIN_EMBEDDINGS_DIR = "../../emb2heights/data/gee_emb_aligned_v2"
TRAIN_TARGETS_DIR = "../../emb2heights/data/patches_labels_10m/"

BATCH_SIZE = 32
PATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4  # L2 Regularization
VAL_SPLIT = 0.2
LAMBDAS = [1.0, 0.5, 0.5, 2.0]  # [MAE, SSIM, Gradient, Structure/Tversky]
RANDOM_SEED = 42
MODEL_TYPE = "lightunet"  # one of: auto, lightunet, decoder_residual

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def save_experiment_config():
    """Logs all hyperparameters to a text file in the experiment folder."""
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

    with open(CONFIG_LOG_PATH, "w") as f:
        f.write(f"--- EXPERIMENT: {EXPERIMENT_NAME} ---\n")
        f.write(f"OUTPUT_DIR: {BASE_DIR}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"PATCH_SIZE: {PATCH_SIZE}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"WEIGHT_DECAY: {WEIGHT_DECAY}\n")
        f.write(f"LOSS LAMBDAS: {LAMBDAS}\n")
        f.write(f"MODEL_TYPE: {MODEL_TYPE}\n")
        f.write(f"TRAIN_EMBEDDINGS_DIR: {TRAIN_EMBEDDINGS_DIR}\n")
        f.write(f"TRAIN_TARGETS_DIR: {TRAIN_TARGETS_DIR}\n")
        f.write(f"VAL_SPLIT: {VAL_SPLIT}\n")
        f.write(f"OPTIMIZER: AdamW\n")
        f.write(f"SCHEDULER: ReduceLROnPlateau (factor=0.5, patience=2)\n")
        f.write(f"GRADIENT CLIPPING: max_norm=1.0\n")
    print(f"📁 Created experiment folder: {EXP_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train emb2heights baseline models")
    parser.add_argument("--model-type", type=str, default=MODEL_TYPE, choices=["auto", "lightunet", "decoder_residual"])
    parser.add_argument("--output-dir", type=str, default=BASE_DIR)
    parser.add_argument("--train-embeddings-dir", type=str, default=TRAIN_EMBEDDINGS_DIR)
    parser.add_argument("--train-targets-dir", type=str, default=TRAIN_TARGETS_DIR)
    parser.add_argument("--experiment-name", type=str, default=EXPERIMENT_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()



def visualize_results(model, dataset, num_samples=3):
    """Generates sample visualizations from the dataset."""
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    target_names = ["% Building", "% Vegetation", "% Water", "nDSM Height (m)"]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, target_tensor = dataset[idx]
            input_batch = img_tensor.unsqueeze(0).to(DEVICE)
            output_batch = model(input_batch)

            target_batch = align_target_to_output(target_tensor.unsqueeze(0).to(DEVICE), output_batch)

            pred = output_batch.squeeze().cpu().numpy()
            true = target_batch.squeeze().cpu().numpy()

            # UN-NORMALIZE HEIGHT FOR VISUALIZATION
            pred[3] = pred[3] * HEIGHT_NORM_CONSTANT
            true[3] = true[3] * HEIGHT_NORM_CONSTANT

            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            for c in range(4):
                vmin, vmax = (0, 1) if c < 3 else (0, HEIGHT_NORM_CONSTANT)
                axes[0, c].imshow(true[c], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0, c].set_title(f"True {target_names[c]}")
                axes[0, c].axis('off')

                axes[1, c].imshow(pred[c], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1, c].set_title(f"Pred {target_names[c]}")
                axes[1, c].axis('off')

            plt.suptitle(f"{model.__class__.__name__} Prediction (Sample {i})")
            plt.tight_layout()
            plt.savefig(os.path.join(VIZ_OUTPUT_DIR, f"viz_{i}.png"))
            plt.close()

# Add an evaluation function to evaluate the model
def binary_iou_from_channel(pred, target, threshold=0.1, eps=1e-6):
    """
    pred, target: [B, H, W]
    """
    pred_mask = pred > threshold
    target_mask = target > threshold

    intersection = (pred_mask & target_mask).sum().float()
    union = (pred_mask | target_mask).sum().float()

    if union == 0:
        return torch.tensor(float("nan"), device=pred.device)

    return (intersection + eps) / (union + eps)


def masked_rmse(pred_height, true_height, mask, eps=1e-6):
    """
    pred_height, true_height: [B, H, W]
    mask: [B, H, W]
    """
    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=pred_height.device)

    return torch.sqrt(torch.mean((pred_height[mask] - true_height[mask]) ** 2))


def evaluate_challenge_metrics(model, val_loader, device, threshold=0.1):
    model.eval()
    debug_printed = False
    building_ious = []
    vegetation_ious = []
    water_ious = []
    building_rmses = []
    vegetation_rmses = []

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)

            # outputs and targets should be [B, 4, H, W]
            pred_building = torch.clamp(outputs[:, 0, :, :], 0, 1)
            pred_veg = torch.clamp(outputs[:, 1, :, :], 0, 1)
            pred_water = torch.clamp(outputs[:, 2, :, :], 0, 1)

            true_building = torch.clamp(targets[:, 0, :, :], 0, 1)
            true_veg = torch.clamp(targets[:, 1, :, :], 0, 1)
            true_water = torch.clamp(targets[:, 2, :, :], 0, 1)

            pred_height = outputs[:, 3, :, :] * HEIGHT_NORM_CONSTANT
            true_height = targets[:, 3, :, :] * HEIGHT_NORM_CONSTANT

            if not debug_printed:
                print("Pred building min/max:", pred_building.min().item(), pred_building.max().item())
                print("True building min/max:", true_building.min().item(), true_building.max().item())
                print("Pred building > threshold:", (pred_building > threshold).sum().item())
                print("True building > threshold:", (true_building > threshold).sum().item())
                debug_printed = True
        
            building_ious.append(
                binary_iou_from_channel(pred_building, true_building, threshold)
            )
            vegetation_ious.append(
                binary_iou_from_channel(pred_veg, true_veg, threshold)
            )
            water_ious.append(
                binary_iou_from_channel(pred_water, true_water, threshold)
            )

            building_mask = true_building > threshold
            vegetation_mask = true_veg > threshold

            building_rmses.append(
                masked_rmse(pred_height, true_height, building_mask)
            )
            vegetation_rmses.append(
                masked_rmse(pred_height, true_height, vegetation_mask)
            )

    metrics = {
        "iou_building": torch.nanmean(torch.stack(building_ious)).item(),
        "iou_vegetation": torch.nanmean(torch.stack(vegetation_ious)).item(),
        "iou_water": torch.nanmean(torch.stack(water_ious)).item(),
        "rmse_building": torch.nanmean(torch.stack(building_rmses)).item(),
        "rmse_vegetation": torch.nanmean(torch.stack(vegetation_rmses)).item(),
    }

    return metrics
# Main Function
def main():
    global BASE_DIR, EXPERIMENT_NAME, EXP_DIR, VIZ_OUTPUT_DIR
    global BEST_MODEL_PATH, LAST_MODEL_PATH, LOSS_CURVE_PATH, CONFIG_LOG_PATH
    global TRAIN_EMBEDDINGS_DIR, TRAIN_TARGETS_DIR, TEST_TARGETS_DIR
    global MODEL_TYPE, EPOCHS, BATCH_SIZE, PATCH_SIZE

    args = parse_args()
    MODEL_TYPE = args.model_type
    BASE_DIR = args.output_dir
    TRAIN_EMBEDDINGS_DIR = args.train_embeddings_dir
    TRAIN_TARGETS_DIR = args.train_targets_dir
    EXPERIMENT_NAME = args.experiment_name
    BATCH_SIZE = args.batch_size
    PATCH_SIZE = args.patch_size
    EPOCHS = args.epochs

    alpha_train_embeddings_dir = os.path.join(TRAIN_EMBEDDINGS_DIR, "alphaearth_emb")
    tessera_train_embeddings_dir = os.path.join(TRAIN_EMBEDDINGS_DIR, "tessera_emb")
    # train_label_dir = os.path.join(TRAIN_TARGETS_DIR, "labels")

    EXP_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)
    VIZ_OUTPUT_DIR = os.path.join(EXP_DIR, "visualizations")
    BEST_MODEL_PATH = os.path.join(EXP_DIR, "model_best_e1.pth")
    LAST_MODEL_PATH = os.path.join(EXP_DIR, "model_last.pth")
    LOSS_CURVE_PATH = os.path.join(EXP_DIR, "loss_curve.png")
    CONFIG_LOG_PATH = os.path.join(EXP_DIR, "training_params.txt")

    save_experiment_config()

    print("--- 1. Data Setup ---")
    # all_train_pairs = find_file_pairs(TRAIN_EMBEDDINGS_DIR, TRAIN_TARGETS_DIR)
    all_train_pairs = find_pixel_fusion_pairs(alpha_train_embeddings_dir, tessera_train_embeddings_dir, TRAIN_TARGETS_DIR)
    print(f"Found {len(all_train_pairs)} training pairs")
    if len(all_train_pairs) == 0:
        raise ValueError(
            "No training (embedding, label) pairs found. "
            f"train_embeddings_dir='{TRAIN_EMBEDDINGS_DIR}', "
            f"train_targets_dir='{TRAIN_TARGETS_DIR}'. "
            "Check filename conventions and directory paths."
        )
    train_pairs, val_pairs = train_test_split(
        all_train_pairs, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

        # In train.py:
    if MODEL_TYPE == "lightunet":
        # train_ds = PixelEmbeddingDataset(train_pairs, patch_size=PATCH_SIZE, is_train=True)
        # val_ds = PixelEmbeddingDataset(val_pairs, patch_size=PATCH_SIZE, is_train=False)
        train_ds = PixelFusionDataset(train_pairs, patch_size=PATCH_SIZE, is_train=True)
        val_ds = PixelFusionDataset(val_pairs, patch_size=PATCH_SIZE, is_train=False)
    else:
        # For the decoders (TerraMind/Thor)
        train_ds = LatentTokenDataset(train_pairs, patch_size=PATCH_SIZE, scale_factor=16, is_train=True)
        val_ds = LatentTokenDataset(val_pairs, patch_size=PATCH_SIZE, scale_factor=16, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)

    sample_img, _ = train_ds[0]
    print(f"Sample image shape: {sample_img.shape}")
    n_channels, n_classes = sample_img.shape[0], 4

    print("--- 2. Model Init ---")
    model, selected_model = build_model(MODEL_TYPE, n_channels, n_classes)
    model = model.to(DEVICE)
    print(f"Using model: {selected_model} (input channels={n_channels})")

    # NEW: AdamW with Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # NEW: Aggressive Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    # criterion = ImprovedCompositeLoss(lambdas=LAMBDAS).to(DEVICE)
    ## L1 Loss (MAE) to test
    criterion = torch.nn.L1Loss().to(DEVICE)

    print(f"Starting training on {DEVICE}...")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_samples_seen = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]", leave=False)
        for imgs, targets in train_pbar:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)

            # loss, l_mae, l_ssim, l_grad, l_tversky = criterion(outputs, targets)
            l_mae = criterion(outputs, targets)
            l_mae.backward()

            # NEW: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += l_mae.item() * imgs.size(0)
            train_samples_seen += imgs.size(0)
            train_avg = running_loss / max(1, train_samples_seen)
            train_pbar.set_postfix(loss=f"{l_mae.item():.4f}", avg=f"{train_avg:.4f}")

        epoch_loss = running_loss / len(train_ds)
        train_losses.append(epoch_loss)

        # --- VALIDATION LOOP ---
        model.eval()
        val_running_loss = 0.0
        val_components = torch.zeros(4).to(DEVICE)
        val_samples_seen = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [val]", leave=False)
            for imgs, targets in val_pbar:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                outputs = model(imgs)

                # loss, l_mae, l_ssim, l_grad, l_tversky = criterion (outputs, targets)
                l_mae = criterion(outputs, targets)
                val_running_loss += l_mae.item() * imgs.size(0)

                bs = imgs.size(0)
                val_components[0] += l_mae * bs
                # val_components[1] += l_ssim * bs
                # val_components[2] += l_grad * bs
                # val_components[3] += l_tversky * bs
                val_samples_seen += bs
                val_avg_live = val_running_loss / max(1, val_samples_seen)
                val_pbar.set_postfix(avg=f"{val_avg_live:.4f}")

        epoch_val_loss = val_running_loss / len(val_ds)
        # epoch_comp = val_components / len(val_ds)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   >> Model Saved! (New Best Val Loss: {best_val_loss:.4f})")

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train: {epoch_loss:.4f} | Val: {epoch_val_loss:.4f}")
        # print(f"   >> Val Breakdown: MAE:{epoch_comp[0]:.3f} | SSIM:{epoch_comp[1]:.3f} | Grad:{epoch_comp[2]:.3f} | Tversky:{epoch_comp[3]:.3f}")
        print(f"   >> Val MAE: {epoch_val_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            metrics = evaluate_challenge_metrics(
                model=model,
                val_loader=val_loader,
                device=DEVICE,
                threshold=0.1)
            
            print("   >> Challenge-style Evaluation")
            print(f"      Building IoU:    {metrics['iou_building']:.4f}")
            print(f"      Vegetation IoU:  {metrics['iou_vegetation']:.4f}")
            print(f"      Water IoU:       {metrics['iou_water']:.4f}")
            print(f"      Building RMSE:   {metrics['rmse_building']:.4f}")
            print(f"      Vegetation RMSE: {metrics['rmse_vegetation']:.4f}")

    print("--- 3. Saving & Visualizing ---")
    torch.save(model.state_dict(), LAST_MODEL_PATH)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f"Training Loss Curve ({EXPERIMENT_NAME})")
    plt.legend()
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()

if __name__ == "__main__":
    main()



