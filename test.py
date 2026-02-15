

"""
Segmentation Validation Script (v3 — Class-Balanced + High-Resolution)
=======================================================================
Evaluates a trained Mask2Former-style segmentation head (v3) on validation /
test data and saves predictions + metrics.

All model definitions are duplicated from chetan_train_segmentation.py v3
so this file is fully self-contained.

Must be used with weights produced by chetan_train_segmentation.py v3.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

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
    img = np.clip(img, 0, 255).astype(np.uint8)
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

class_names = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky",
]

n_classes = len(value_map)

color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


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

        return image, mask, data_id


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
# Residual Conv Block
# ============================================================================

class ResidualConvBlock(nn.Module):
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


# ============================================================================
# High-Resolution Feature Pyramid Decoder (Dual Output)
# ============================================================================

class HighResFeaturePyramidDecoder(nn.Module):
    """
    FPN pixel decoder — dual output:
      1. lowres_features  (tokenH×tokenW)    — for cross-attention (memory safe)
      2. highres_features (tokenH*8×tokenW*8) — for mask prediction (sharp masks)
    """

    def __init__(self, in_channels, hidden_dim=256, tokenH=28, tokenW=48, dropout=0.1):
        super().__init__()
        self.tokenH = tokenH
        self.tokenW = tokenW

        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
            )
            for _ in range(4)
        ])

        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Dropout2d(dropout),
            )
            for _ in range(4)
        ])

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

        spatial = []
        for feat in multi_scale_features:
            C = feat.shape[-1]
            spatial.append(
                feat.reshape(B, self.tokenH, self.tokenW, C).permute(0, 3, 1, 2)
            )

        laterals = [lat(s) for lat, s in zip(self.laterals, spatial)]

        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:],
                mode="bilinear", align_corners=False,
            )

        fpn_features = [smooth(lat) for smooth, lat in zip(self.smooth, laterals)]

        # LOW-RES: fused FPN at tokenH×tokenW
        lowres_features = sum(fpn_features)

        # HIGH-RES: progressive upsample
        x = self.up_stage1(lowres_features)
        x = self.refine1(x)
        x = self.up_stage2(x)
        x = self.refine2(x)
        x = self.up_stage3(x)
        x = self.refine3(x)
        highres_features = x

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
# Mask2Former Head — Dual Resolution
# ============================================================================

class Mask2FormerHead(nn.Module):
    """
    Mask2Former head with memory-efficient dual-resolution design:
      • Cross-attention on LOW-RES FPN features (tokenH×tokenW = 1344)
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

        # ---- Pixel Decoder ----
        self.pixel_decoder = HighResFeaturePyramidDecoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            tokenH=tokenH,
            tokenW=tokenW,
            dropout=dropout,
        )

        # Project LOW-RES features for cross-attention
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
        highres_dim = self.pixel_decoder.highres_channels  # 64
        lowres_dim = self.pixel_decoder.lowres_channels    # 256

        # HIGH-RES mask embed (for final mask prediction)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, highres_dim),
        )

        # LOW-RES mask embed (for attention masking — memory efficient)
        self.lowres_mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, lowres_dim),
        )

        # Class prediction
        self.class_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes + 1),
        )

        # ---- Mask Refiner ----
        self.mask_refiner = MaskRefiner(hidden_channels=32, dropout=dropout)

    def _predict_highres(self, queries, highres_features):
        class_logits = self.class_embed(queries)
        mask_embeds = self.mask_embed(queries)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeds, highres_features)
        return {"pred_logits": class_logits, "pred_masks": pred_masks}

    def _predict_lowres(self, queries, lowres_features):
        mask_embeds = self.lowres_mask_embed(queries)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeds, lowres_features)
        return pred_masks

    def forward(self, multi_scale_features):
        B = multi_scale_features[0].shape[0]

        # ---- Pixel decoder: dual output ----
        highres_features, lowres_features, fpn_features = self.pixel_decoder(multi_scale_features)

        # Cross-attention memory: low-res
        lowres_embed = self.lowres_proj(lowres_features)
        pixel_memory = lowres_embed.flatten(2).permute(0, 2, 1)

        # ---- Initialize queries ----
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # ---- Iterative decoding ----
        aux_outputs = []
        attn_mask = None

        for layer_idx, layer in enumerate(self.decoder_layers):
            queries = layer(queries, pixel_memory, attn_mask=attn_mask)

            layer_queries = self.decoder_norm(queries)

            # HIGH-RES prediction for deep supervision
            aux_pred = self._predict_highres(layer_queries, highres_features)
            aux_outputs.append(aux_pred)

            # LOW-RES masks for attention masking (memory efficient)
            lowres_masks = self._predict_lowres(layer_queries, lowres_features)
            flat_masks = lowres_masks.flatten(2)
            attn_mask = (flat_masks.sigmoid() < 0.5)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
            attn_mask = attn_mask.reshape(B * self.nhead, self.num_queries, -1)
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
    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    dice_per_class = []
    for c in range(num_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum().float()
        dice = (2.0 * inter + smooth) / (p.sum().float() + t.sum().float() + smooth)
        dice_per_class.append(dice.item())
    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    return (torch.argmax(pred, dim=1) == target).float().mean().item()


# ============================================================================
# Visualization
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);       axes[0].set_title("Input Image"); axes[0].axis("off")
    axes[1].imshow(gt_color);  axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(pred_color); axes[2].set_title("Prediction");  axes[2].axis("off")
    plt.suptitle(f"Sample: {data_id}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_metrics_summary(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(filepath, "w") as f:
        f.write("EVALUATION RESULTS (Mask2Former v3 — Dual-Res, Class-Balanced)\n")
        f.write("=" * 60 + "\n\n")

        f.write("Overall Metrics:\n")
        f.write(f"  Mean IoU:            {results['mean_iou']:.4f}\n")
        f.write(f"  Mean Dice:           {results['mean_dice']:.4f}\n")
        f.write(f"  Mean Pixel Accuracy: {results['mean_pixel_acc']:.4f}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for name, iou in zip(class_names, results["class_iou"]):
            val = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {val}\n")
        f.write("\n")

        f.write("Per-Class Dice:\n")
        f.write("-" * 40 + "\n")
        for name, dice in zip(class_names, results["class_dice"]):
            val = f"{dice:.4f}" if not np.isnan(dice) else "N/A"
            f.write(f"  {name:<20}: {val}\n")

    print(f"\nSaved evaluation metrics to {filepath}")

    # Bar charts
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    valid_iou = [v if not np.isnan(v) else 0 for v in results["class_iou"]]
    axes[0].bar(range(n_classes), valid_iou,
                color=[color_palette[i] / 255.0 for i in range(n_classes)], edgecolor="black")
    axes[0].set_xticks(range(n_classes))
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].set_ylabel("IoU")
    axes[0].set_title(f"Per-Class IoU (Mean: {results['mean_iou']:.4f})")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=results["mean_iou"], color="red", linestyle="--", label="Mean")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    valid_dice = [v if not np.isnan(v) else 0 for v in results["class_dice"]]
    axes[1].bar(range(n_classes), valid_dice,
                color=[color_palette[i] / 255.0 for i in range(n_classes)], edgecolor="black")
    axes[1].set_xticks(range(n_classes))
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title(f"Per-Class Dice (Mean: {results['mean_dice']:.4f})")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(y=results["mean_dice"], color="red", linestyle="--", label="Mean")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Mask2Former v3 segmentation evaluation")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(script_dir, "potato2_segmentation_head_best.pth"))
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(script_dir, "..", "Offroad_Segmentation_testImages"))
    parser.add_argument("--output_dir", type=str, default="./potatoFINAL_predictions")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=5)

    # Resolution (must match training)
    parser.add_argument("--height", type=int, default=392)
    parser.add_argument("--width", type=int, default=672)

    # Architecture (must match training)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    assert args.height % 14 == 0, f"Height {args.height} must be a multiple of 14"
    assert args.width % 14 == 0, f"Width {args.width} must be a multiple of 14"

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    h, w = args.height, args.width
    tokenH, tokenW = h // 14, w // 14
    print(f"Input resolution: {w}×{h} (tokens: {tokenW}×{tokenH})")

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

    # ----- Dataset -----
    print(f"Loading dataset from {args.data_dir}...")
    valset = MaskDataset(data_dir=args.data_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"Loaded {len(valset)} samples")

    # ----- DINOv2 backbone -----
    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14", "base": "vitb14_reg",
        "large": "vitl14_reg", "giant": "vitg14_reg",
    }
    backbone_name = f"dinov2_{backbone_archs[BACKBONE_SIZE]}"
    raw_backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    raw_backbone.eval()
    raw_backbone.to(device)

    backbone = DINOv2MultiScaleWrapper(raw_backbone, n_last_layers=4).to(device)
    backbone.eval()

    sample_img, _, _ = valset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    with torch.no_grad():
        test_features = backbone(sample_img)
    n_embedding = test_features[0].shape[2]
    print(f"Backbone loaded! embed_dim={n_embedding}, {len(test_features)} scale levels")
    del test_features

    # ----- Load model -----
    print(f"Loading Mask2Former head from {args.model_path}...")
    model = Mask2FormerHead(
        in_channels=n_embedding,
        num_classes=n_classes,
        tokenH=tokenH,
        tokenW=tokenW,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        nhead=args.nhead,
        dropout=args.dropout,
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded! Total parameters: {total_params:,}")

    # ----- Output dirs -----
    masks_dir = os.path.join(args.output_dir, "masks")
    masks_color_dir = os.path.join(args.output_dir, "masks_color")
    comparisons_dir = os.path.join(args.output_dir, "comparisons")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # ================================================================
    # Evaluate
    # ================================================================
    print(f"\nRunning evaluation on {len(valset)} images...")

    iou_scores, dice_scores, pixel_accuracies = [], [], []
    all_class_iou, all_class_dice = [], []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", unit="batch")
        for batch_idx, (imgs, labels, data_ids) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            multi_feats = backbone(imgs)
            outputs = model(multi_feats)
            sem_seg = mask2former_inference(outputs, img_size=imgs.shape[2:], num_classes=n_classes)

            labels_squeezed = labels.squeeze(1).long()
            predicted_masks = torch.argmax(sem_seg, dim=1)

            iou, class_iou = compute_iou(sem_seg, labels_squeezed, num_classes=n_classes)
            dice, class_dice = compute_dice(sem_seg, labels_squeezed, num_classes=n_classes)
            pixel_acc = compute_pixel_accuracy(sem_seg, labels_squeezed)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_iou.append(class_iou)
            all_class_dice.append(class_dice)

            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                pred_np = predicted_masks[i].cpu().numpy().astype(np.uint8)
                Image.fromarray(pred_np).save(os.path.join(masks_dir, f"{base_name}_pred.png"))

                pred_color = mask_to_color(pred_np)
                cv2.imwrite(
                    os.path.join(masks_color_dir, f"{base_name}_pred_color.png"),
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR),
                )

                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_squeezed[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f"sample_{sample_count}_comparison.png"),
                        data_id,
                    )
                sample_count += 1

            pbar.set_postfix(iou=f"{iou:.3f}", dice=f"{dice:.3f}", acc=f"{pixel_acc:.3f}")

    # ---- Results ----
    mean_iou = np.nanmean(iou_scores)
    mean_dice = np.nanmean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)
    avg_class_iou = np.nanmean(all_class_iou, axis=0)
    avg_class_dice = np.nanmean(all_class_dice, axis=0)

    results = {
        "mean_iou": mean_iou, "mean_dice": mean_dice, "mean_pixel_acc": mean_pixel_acc,
        "class_iou": avg_class_iou, "class_dice": avg_class_dice,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Mean IoU:            {mean_iou:.4f}")
    print(f"  Mean Dice:           {mean_dice:.4f}")
    print(f"  Mean Pixel Accuracy: {mean_pixel_acc:.4f}")
    print("=" * 60)

    print("\nPer-Class IoU:")
    for name, iou in zip(class_names, avg_class_iou):
        print(f"  {name:<20}: {iou:.4f}" if not np.isnan(iou) else f"  {name:<20}: N/A")

    print("\nPer-Class Dice:")
    for name, dice in zip(class_names, avg_class_dice):
        print(f"  {name:<20}: {dice:.4f}" if not np.isnan(dice) else f"  {name:<20}: N/A")

    save_metrics_summary(results, args.output_dir)

    print(f"\nEvaluation complete! Processed {len(valset)} images.")
    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  - masks/                  : Raw prediction masks (class IDs 0-9)")
    print(f"  - masks_color/            : Colored prediction masks (RGB)")
    print(f"  - comparisons/            : Side-by-side comparisons ({args.num_samples} samples)")
    print(f"  - evaluation_metrics.txt  : Full text report")
    print(f"  - per_class_metrics.png   : IoU + Dice bar charts")


if __name__ == "__main__":
    main()
