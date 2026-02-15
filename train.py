"""
Segmentation Training Script (v3 — Class-Balanced + High-Resolution)
=====================================================================
Trains a Mask2Former-style segmentation head on top of a frozen DINOv2
backbone with:
  • Inverse-frequency class weighting (CE + Dice)
  • Focal modulation on cross-entropy
  • Boundary-aware loss for sharp edges
  • Deep supervision / auxiliary losses from intermediate decoder layers
  • Higher input resolution (¾ scale instead of ½)
  • 8× progressive upsampling pixel decoder for fine-grained masks
  • Mask refinement CNN for crisp boundary prediction
  • Memory-efficient: cross-attention at FPN resolution, masks at 8× resolution
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
from tqdm import tqdm
from collections import Counter

plt.switch_backend("Agg")


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9,    # Sky
}
n_classes = len(value_map)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Class Weight Computation
# ============================================================================

def compute_class_weights(dataset, num_classes, device, method="inv_freq_log"):
    """
    Scan raw mask files on disk to count per-class pixel frequencies.
    This runs entirely on CPU with raw PIL images — zero GPU memory.

    Parameters
    ----------
    dataset : MaskDataset
        Only used to read .masks_dir and .data_ids paths.
    num_classes : int
    device : torch.device
        Weights are moved here at the end.
    method : str
        'inv_freq'     — 1 / count
        'inv_freq_log' — 1 / log(1.02 + freq)  (recommended)
        'median_freq'  — median_freq / class_freq
    """
    print("Scanning dataset for class frequencies (CPU-only, no GPU memory)...")
    class_pixel_counts = np.zeros(num_classes, dtype=np.float64)

    for idx in tqdm(range(len(dataset.data_ids)), desc="Counting pixels", leave=False):
        data_id = dataset.data_ids[idx]
        mask_path = os.path.join(dataset.masks_dir, data_id)

        # Read raw mask and convert to class IDs — pure PIL + numpy, no transforms
        raw_mask = Image.open(mask_path)
        mask_np = np.array(convert_mask(raw_mask), dtype=np.uint8)

        for c in range(num_classes):
            class_pixel_counts[c] += (mask_np == c).sum()

    total_pixels = class_pixel_counts.sum()
    freqs = class_pixel_counts / total_pixels

    class_names_local = [
        "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
        "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky",
    ]

    print("\nClass distribution:")
    for name, count, freq in zip(class_names_local, class_pixel_counts, freqs):
        print(f"  {name:<20}: {int(count):>12,} pixels  ({freq * 100:.2f}%)")

    # Compute weights
    freqs_t = torch.from_numpy(freqs).float()

    if method == "inv_freq":
        weights = 1.0 / (freqs_t + 1e-10)
    elif method == "inv_freq_log":
        weights = 1.0 / torch.log(1.02 + freqs_t)
    elif method == "median_freq":
        nonzero = freqs_t[freqs_t > 0]
        median_freq = nonzero.median()
        weights = median_freq / (freqs_t + 1e-10)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to mean=1, clamp for stability
    weights = weights / weights.mean()
    weights = weights.clamp(min=0.1, max=10.0)

    print("\nComputed class weights:")
    for name, w in zip(class_names_local, weights):
        print(f"  {name:<20}: {w:.4f}")

    return weights.to(device)


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.masks_dir = os.path.join(data_dir, "Segmentation")
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask


# ============================================================================
# Multi-Scale Feature Extraction
# ============================================================================

class DINOv2MultiScaleWrapper(nn.Module):
    def __init__(self, backbone, n_last_layers=4):
        super().__init__()
        self.backbone = backbone
        self.n_last_layers = n_last_layers
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        features = self.backbone.get_intermediate_layers(
            x, n=self.n_last_layers, reshape=False,
        )
        return list(features)


# ============================================================================
# High-Resolution Feature Pyramid Decoder
# ============================================================================

class ResidualConvBlock(nn.Module):
    """Residual block with BN for feature refinement at each upsample stage."""

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class HighResFeaturePyramidDecoder(nn.Module):
    """
    FPN pixel decoder that produces TWO outputs:
      1. Low-res FPN features (tokenH×tokenW) for cross-attention — memory efficient
      2. High-res pixel features (8× upsampled) for mask prediction ��� sharp masks

    This dual-path design avoids materializing huge attention masks.
    """

    def __init__(self, in_channels, hidden_dim=256, tokenH=28, tokenW=48, dropout=0.1):
        super().__init__()
        self.tokenH = tokenH
        self.tokenW = tokenW

        # Lateral projections (one per scale level)
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
            )
            for _ in range(4)
        ])

        # Smoothing after top-down fusion
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Dropout2d(dropout),
            )
            for _ in range(4)
        ])

        # ---- High-res path: 3-stage progressive upsample for MASK prediction ----
        # Stage 1: hidden_dim → hidden_dim, 2×
        self.up_stage1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.refine1 = ResidualConvBlock(hidden_dim, dropout)

        # Stage 2: hidden_dim → hidden_dim//2, 2×
        self.up_stage2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
        )
        self.refine2 = ResidualConvBlock(hidden_dim // 2, dropout)

        # Stage 3: hidden_dim//2 → hidden_dim//4, 2×
        self.up_stage3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.GELU(),
        )
        self.refine3 = ResidualConvBlock(hidden_dim // 4, dropout)

        self.highres_channels = hidden_dim // 4  # 64
        self.lowres_channels = hidden_dim         # 256

    def forward(self, multi_scale_features):
        B = multi_scale_features[0].shape[0]

        # Reshape [B, N, C] → [B, C, H, W]
        spatial = []
        for feat in multi_scale_features:
            C = feat.shape[-1]
            spatial.append(
                feat.reshape(B, self.tokenH, self.tokenW, C).permute(0, 3, 1, 2)
            )

        # Lateral projections
        laterals = [lat(s) for lat, s in zip(self.laterals, spatial)]

        # Top-down fusion
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:],
                mode="bilinear", align_corners=False,
            )

        # Smooth
        fpn_features = [smooth(lat) for smooth, lat in zip(self.smooth, laterals)]

        # LOW-RES output: fused FPN at tokenH×tokenW for cross-attention
        lowres_features = sum(fpn_features)  # [B, hidden_dim, tokenH, tokenW]

        # HIGH-RES output: progressive upsample for mask prediction
        x = self.up_stage1(lowres_features)
        x = self.refine1(x)
        x = self.up_stage2(x)
        x = self.refine2(x)
        x = self.up_stage3(x)
        x = self.refine3(x)
        highres_features = x  # [B, hidden_dim//4, tokenH*8, tokenW*8]

        return highres_features, lowres_features, fpn_features


# ============================================================================
# Mask Refinement Network
# ============================================================================

class MaskRefiner(nn.Module):
    """
    Lightweight CNN that refines per-query mask logits for sharper boundaries.
    """

    def __init__(self, hidden_channels=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, masks):
        """masks: [B, Q, H, W] → refined [B, Q, H, W]"""
        B, Q, H, W = masks.shape
        x = masks.reshape(B * Q, 1, H, W)
        residual = self.net(x)
        refined = x + residual
        return refined.reshape(B, Q, H, W)


# ============================================================================
# Transformer Decoder with Masked Attention
# ============================================================================

class MaskedTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, pixel_memory, attn_mask=None):
        q = self.norm1(queries)
        attended, _ = self.cross_attn(
            query=q, key=pixel_memory, value=pixel_memory, attn_mask=attn_mask,
        )
        queries = queries + self.dropout1(attended)

        q2 = self.norm2(queries)
        self_attended, _ = self.self_attn(query=q2, key=q2, value=q2)
        queries = queries + self.dropout2(self_attended)

        queries = queries + self.ffn(self.norm3(queries))
        return queries


# ============================================================================
# Mask2Former Head — Dual Resolution (Low-Res Attention + High-Res Masks)
# ============================================================================

class Mask2FormerHead(nn.Module):
    """
    Mask2Former head with memory-efficient dual-resolution design:
      • Cross-attention on LOW-RES FPN features (tokenH×tokenW = 28×48 = 1344)
      • Mask prediction on HIGH-RES pixel features (8× = 224×384)
      • Deep supervision from intermediate decoder layers
      • Mask refinement CNN on final output
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        tokenH,
        tokenW,
        hidden_dim=256,
        num_queries=100,
        num_decoder_layers=3,
        nhead=8,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.nhead = nhead
        self.tokenH = tokenH
        self.tokenW = tokenW
        self.num_decoder_layers = num_decoder_layers

        # ---- Pixel Decoder (produces both low-res and high-res features) ----
        self.pixel_decoder = HighResFeaturePyramidDecoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            tokenH=tokenH,
            tokenW=tokenW,
            dropout=dropout,
        )

        # Project LOW-RES features to query dim for cross-attention
        # lowres is already hidden_dim, so just a light refinement
        self.lowres_proj = nn.Sequential(
            nn.Conv2d(self.pixel_decoder.lowres_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # ---- Learnable object queries ----
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # ---- Transformer decoder layers ----
        self.decoder_layers = nn.ModuleList([
            MaskedTransformerDecoderLayer(
                d_model=hidden_dim, nhead=nhead,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # ---- Prediction heads ----
        # Mask embed projects to HIGH-RES channels for dot product with highres features
        highres_dim = self.pixel_decoder.highres_channels  # 64

        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, highres_dim),
        )

        # For low-res intermediate mask predictions (used for attention masking)
        lowres_dim = self.pixel_decoder.lowres_channels  # 256
        self.lowres_mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, lowres_dim),
        )

        self.class_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes + 1),
        )

        # ---- Mask Refiner (applied to final high-res prediction) ----
        self.mask_refiner = MaskRefiner(hidden_channels=32, dropout=dropout)

    def _predict_highres(self, queries, highres_features):
        """Predict masks at HIGH resolution for output."""
        class_logits = self.class_embed(queries)
        mask_embeds = self.mask_embed(queries)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeds, highres_features)
        return {"pred_logits": class_logits, "pred_masks": pred_masks}

    def _predict_lowres(self, queries, lowres_features):
        """Predict masks at LOW resolution for attention masking."""
        mask_embeds = self.lowres_mask_embed(queries)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeds, lowres_features)
        return pred_masks

    def forward(self, multi_scale_features):
        B = multi_scale_features[0].shape[0]

        # ---- Pixel decoder: dual output ----
        highres_features, lowres_features, fpn_features = self.pixel_decoder(multi_scale_features)
        # highres_features: [B, 64, tokenH*8, tokenW*8]  — for mask prediction
        # lowres_features:  [B, 256, tokenH, tokenW]     — for cross-attention

        # Cross-attention memory: low-res
        lowres_embed = self.lowres_proj(lowres_features)  # [B, hidden_dim, tokenH, tokenW]
        pixel_memory = lowres_embed.flatten(2).permute(0, 2, 1)  # [B, tokenH*tokenW, hidden_dim]
        # tokenH*tokenW = 28*48 = 1344 — very manageable for attention

        # ---- Initialize queries ----
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # ---- Iterative decoding with deep supervision ----
        aux_outputs = []
        attn_mask = None

        for layer_idx, layer in enumerate(self.decoder_layers):
            queries = layer(queries, pixel_memory, attn_mask=attn_mask)

            # Normalized queries for prediction
            layer_queries = self.decoder_norm(queries)

            # HIGH-RES prediction for deep supervision loss
            aux_pred = self._predict_highres(layer_queries, highres_features)
            aux_outputs.append(aux_pred)

            # LOW-RES mask prediction for attention masking (memory efficient)
            lowres_masks = self._predict_lowres(layer_queries, lowres_features)
            flat_masks = lowres_masks.flatten(2)  # [B, Q, tokenH*tokenW]
            attn_mask = (flat_masks.sigmoid() < 0.5)  # True = masked
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
            attn_mask = attn_mask.reshape(B * self.nhead, self.num_queries, -1)
            # Safety: ensure not ALL masked per query
            all_masked = attn_mask.all(dim=-1, keepdim=True)
            attn_mask = attn_mask & ~all_masked

        # ---- Final prediction + mask refinement ----
        final_output = aux_outputs[-1]
        final_output["pred_masks"] = self.mask_refiner(final_output["pred_masks"])

        return {
            "pred_logits": final_output["pred_logits"],
            "pred_masks": final_output["pred_masks"],
            "aux_outputs": aux_outputs[:-1],
        }


def mask2former_inference(outputs, img_size, num_classes):
    """Convert per-query predictions to dense class-probability map."""
    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]

    pred_masks = F.interpolate(
        pred_masks, size=img_size, mode="bilinear", align_corners=False,
    )

    cls_probs = F.softmax(pred_logits, dim=-1)[..., :-1]
    mask_probs = pred_masks.sigmoid()

    sem_seg = torch.einsum("bqc,bqhw->bchw", cls_probs, mask_probs)
    return sem_seg


# ============================================================================
# Loss: Weighted Focal CE + Weighted Dice + Boundary + Deep Supervision
# ============================================================================

class BoundaryLoss(nn.Module):
    """
    Computes extra penalty at class boundaries using Laplacian edge detection
    on the ground-truth mask.
    """

    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("laplacian", kernel)

    def forward(self, pred, target, num_classes):
        target_float = target.unsqueeze(1).float()
        edges = F.conv2d(target_float, self.laplacian.to(target.device), padding=1)
        boundary_mask = (edges.abs() > 0).float().squeeze(1)

        ce_map = F.cross_entropy(pred, target, reduction="none")
        boundary_loss = (ce_map * boundary_mask).sum() / (boundary_mask.sum() + 1e-6)

        return boundary_loss


class FocalCELoss(nn.Module):
    """Focal Cross-Entropy with per-class weights."""

    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(
            pred, target, weight=self.class_weights, reduction="none",
        )

        probs = F.softmax(pred, dim=1)
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - target_probs) ** self.gamma

        loss = (focal_weight * ce_loss).mean()
        return loss


class WeightedDiceLoss(nn.Module):
    """Dice loss with per-class weights."""

    def __init__(self, class_weights=None, smooth=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.smooth = smooth

    def forward(self, pred, target, num_classes):
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (pred_soft * target_one_hot).sum(dim=dims)
        cardinality = pred_soft.sum(dim=dims) + target_one_hot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.class_weights is not None:
            w = self.class_weights.to(dice_per_class.device)
            weighted_dice = (w * dice_per_class).sum() / w.sum()
        else:
            weighted_dice = dice_per_class.mean()

        return 1.0 - weighted_dice


class CombinedSegmentationLoss(nn.Module):
    """
    Total = w_focal * FocalCE + w_dice * WeightedDice + w_boundary * BoundaryLoss
    Applied to main output + deep supervision auxiliary outputs.
    """

    def __init__(self, class_weights=None, focal_gamma=2.0,
                 w_focal=0.4, w_dice=0.4, w_boundary=0.2,
                 aux_weight=0.4):
        super().__init__()
        self.focal_ce = FocalCELoss(class_weights=class_weights, gamma=focal_gamma)
        self.dice_loss = WeightedDiceLoss(class_weights=class_weights)
        self.boundary_loss = BoundaryLoss()
        self.w_focal = w_focal
        self.w_dice = w_dice
        self.w_boundary = w_boundary
        self.aux_weight = aux_weight

    def _single_scale_loss(self, pred, target, num_classes):
        l_focal = self.focal_ce(pred, target)
        l_dice = self.dice_loss(pred, target, num_classes)
        l_boundary = self.boundary_loss(pred, target, num_classes)
        return self.w_focal * l_focal + self.w_dice * l_dice + self.w_boundary * l_boundary

    def forward(self, main_sem_seg, target, num_classes, aux_outputs=None, img_size=None):
        loss = self._single_scale_loss(main_sem_seg, target, num_classes)

        if aux_outputs is not None and img_size is not None:
            for aux in aux_outputs:
                aux_sem = mask2former_inference(aux, img_size=img_size, num_classes=num_classes)
                aux_loss = self._single_scale_loss(aux_sem, target, num_classes)
                loss = loss + self.aux_weight * aux_loss

        return loss


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_per_class = []
    for c in range(num_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        iou_per_class.append((inter / union).item() if union > 0 else float("nan"))
    return np.nanmean(iou_per_class)


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    dice_per_class = []
    for c in range(num_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum().float()
        dice = (2.0 * inter + smooth) / (p.sum().float() + t.sum().float() + smooth)
        dice_per_class.append(dice.item())
    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    return (torch.argmax(pred, dim=1) == target).float().mean().item()


# ============================================================================
# Plotting / Saving
# ============================================================================

def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train"); plt.plot(history["val_loss"], label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history["train_pixel_acc"], label="train"); plt.plot(history["val_pixel_acc"], label="val")
    plt.title("Pixel Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "training_curves.png")); plt.close()
    print(f"Saved training curves to '{output_dir}/training_curves.png'")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_iou"], label="Train IoU")
    plt.title("Train IoU"); plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history["val_iou"], label="Val IoU")
    plt.title("Val IoU"); plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "iou_curves.png")); plt.close()
    print(f"Saved IoU curves to '{output_dir}/iou_curves.png'")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_dice"], label="Train Dice")
    plt.title("Train Dice"); plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history["val_dice"], label="Val Dice")
    plt.title("Val Dice"); plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "dice_curves.png")); plt.close()
    print(f"Saved Dice curves to '{output_dir}/dice_curves.png'")

    plt.figure(figsize=(12, 10))
    for i, (key_t, key_v, title) in enumerate([
        ("train_loss", "val_loss", "Loss"),
        ("train_iou", "val_iou", "IoU"),
        ("train_dice", "val_dice", "Dice Score"),
        ("train_pixel_acc", "val_pixel_acc", "Pixel Accuracy"),
    ], 1):
        plt.subplot(2, 2, i)
        plt.plot(history[key_t], label="train"); plt.plot(history[key_v], label="val")
        plt.title(title); plt.xlabel("Epoch"); plt.ylabel(title); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "all_metrics_curves.png")); plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "evaluation_metrics.txt")

    with open(filepath, "w") as f:
        f.write("TRAINING RESULTS (Mask2Former v3 — Class-Balanced + High-Res)\n")
        f.write("=" * 60 + "\n\n")

        f.write("Final Metrics:\n")
        for key in ["train_loss", "val_loss", "train_iou", "val_iou",
                     "train_dice", "val_dice", "train_pixel_acc", "val_pixel_acc"]:
            f.write(f"  {key:>20s}: {history[key][-1]:.4f}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 60 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ["Epoch", "TrLoss", "VaLoss", "TrIoU", "VaIoU",
                    "TrDice", "VaDice", "TrAcc", "VaAcc"]
        f.write("{:<8}{:<12}{:<12}{:<12}{:<12}{:<12}{:<12}{:<12}{:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")
        for i in range(len(history["train_loss"])):
            f.write(
                "{:<8}{:<12.4f}{:<12.4f}{:<12.4f}{:<12.4f}{:<12.4f}{:<12.4f}{:<12.4f}{:<12.4f}\n".format(
                    i + 1,
                    history["train_loss"][i], history["val_loss"][i],
                    history["train_iou"][i], history["val_iou"][i],
                    history["train_dice"][i], history["val_dice"][i],
                    history["train_pixel_acc"][i], history["val_pixel_acc"][i],
                ))
    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- Hyperparameters -----
    batch_size = 2
    h = 392   # 28 * 14
    w = 672   # 48 * 14
    lr = 1e-4
    weight_decay = 0.01
    n_epochs = 10
    num_queries = 100
    num_decoder_layers = 3
    hidden_dim = 256
    dropout = 0.1
    focal_gamma = 2.0

    tokenH = h // 14   # 28
    tokenW = w // 14   # 48

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "potato2_train_stats")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input resolution: {w}×{h} (tokens: {tokenW}×{tokenH})")
    print(f"Cross-attention memory: {tokenH*tokenW} positions (low-res)")
    print(f"Mask prediction: ~{tokenH*8}×{tokenW*8} (high-res)")

    # ----- Transforms -----
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # ----- Datasets -----
    data_dir = os.path.join(script_dir, "..", "Offroad_Segmentation_Training_Dataset", "train")
    val_dir = os.path.join(script_dir, "..", "Offroad_Segmentation_Training_Dataset", "val")

    trainset = MaskDataset(data_dir=data_dir, transform=transform, mask_transform=mask_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    valset = MaskDataset(data_dir=val_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # ----- Compute class weights from training set -----
    class_weights = compute_class_weights(trainset, n_classes, device, method="inv_freq_log")

    # ----- Load DINOv2 backbone -----
    print("\nLoading DINOv2 backbone...")
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_name = f"dinov2_{backbone_archs[BACKBONE_SIZE]}"
    raw_backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    raw_backbone.eval()
    raw_backbone.to(device)

    backbone = DINOv2MultiScaleWrapper(raw_backbone, n_last_layers=4).to(device)
    backbone.eval()

    # Determine embedding dimension
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        test_features = backbone(imgs)
    n_embedding = test_features[0].shape[2]
    print(f"Backbone loaded! embed_dim={n_embedding}, {len(test_features)} scale levels")
    print(f"Each level shape: {test_features[0].shape}")
    del test_features

    # ----- Create Mask2Former head -----
    model = Mask2FormerHead(
        in_channels=n_embedding,
        num_classes=n_classes,
        tokenH=tokenH,
        tokenW=tokenW,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        nhead=8,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Mask2Former head trainable parameters: {total_params:,}")

    # ----- Loss & Optimizer -----
    loss_fn = CombinedSegmentationLoss(
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        w_focal=0.4,
        w_dice=0.4,
        w_boundary=0.2,
        aux_weight=0.4,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # ----- Training history -----
    history = {k: [] for k in [
        "train_loss", "val_loss", "train_iou", "val_iou",
        "train_dice", "val_dice", "train_pixel_acc", "val_pixel_acc",
    ]}
    best_val_iou = 0.0

    # ================================================================
    # Training loop
    # ================================================================
    print(f"\nStarting training for {n_epochs} epochs...")
    print("=" * 80)

    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # ---------- TRAIN ----------
        model.train()
        train_losses, train_ious, train_dices, train_accs = [], [], [], []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]",
                          leave=False, unit="batch")
        for imgs, labels in train_pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).squeeze(1).long()

            with torch.no_grad():
                multi_feats = backbone(imgs)

            outputs = model(multi_feats)
            sem_seg = mask2former_inference(outputs, img_size=imgs.shape[2:], num_classes=n_classes)

            loss = loss_fn(
                sem_seg, labels, n_classes,
                aux_outputs=outputs.get("aux_outputs"),
                img_size=imgs.shape[2:],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                train_losses.append(loss.item())
                train_ious.append(compute_iou(sem_seg, labels, n_classes))
                train_dices.append(compute_dice(sem_seg, labels, n_classes))
                train_accs.append(compute_pixel_accuracy(sem_seg, labels))

            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        val_losses, val_ious, val_dices, val_accs = [], [], [], []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]",
                        leave=False, unit="batch")
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).squeeze(1).long()

                multi_feats = backbone(imgs)
                outputs = model(multi_feats)
                sem_seg = mask2former_inference(outputs, img_size=imgs.shape[2:], num_classes=n_classes)

                loss = loss_fn(
                    sem_seg, labels, n_classes,
                    aux_outputs=outputs.get("aux_outputs"),
                    img_size=imgs.shape[2:],
                )

                val_losses.append(loss.item())
                val_ious.append(compute_iou(sem_seg, labels, n_classes))
                val_dices.append(compute_dice(sem_seg, labels, n_classes))
                val_accs.append(compute_pixel_accuracy(sem_seg, labels))

                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---------- Aggregate & store ----------
        epoch_metrics = {
            "train_loss": np.mean(train_losses),
            "val_loss": np.mean(val_losses),
            "train_iou": np.nanmean(train_ious),
            "val_iou": np.nanmean(val_ious),
            "train_dice": np.mean(train_dices),
            "val_dice": np.mean(val_dices),
            "train_pixel_acc": np.mean(train_accs),
            "val_pixel_acc": np.mean(val_accs),
        }
        for k, v in epoch_metrics.items():
            history[k].append(v)

        scheduler.step()

        if epoch_metrics["val_iou"] > best_val_iou:
            best_val_iou = epoch_metrics["val_iou"]
            best_path = os.path.join(script_dir, "potato2_segmentation_head_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"\n  ★ New best Val IoU: {best_val_iou:.4f} — saved to '{best_path}'")

        epoch_pbar.set_postfix(
            tr_loss=f"{epoch_metrics['train_loss']:.3f}",
            va_loss=f"{epoch_metrics['val_loss']:.3f}",
            va_iou=f"{epoch_metrics['val_iou']:.3f}",
            va_acc=f"{epoch_metrics['val_pixel_acc']:.3f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    # ---- Save everything ----
    print("\nSaving training curves...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    model_path = os.path.join(script_dir, "potato2_segmentation_head_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved final model to '{model_path}'")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Best Val IoU:       {best_val_iou:.4f}")
    print(f"  Final Val Loss:     {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_pixel_acc'][-1]:.4f}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
