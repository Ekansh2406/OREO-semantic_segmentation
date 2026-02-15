# 🌲 Mask2Former v3 — Dual-Resolution Class-Balanced Offroad Segmentation

High-resolution semantic segmentation using a **DINOv2 frozen backbone** + a **dual-resolution Mask2Former-style head**, designed for imbalanced off-road terrain classes.
---

# 📂 Dataset Setup (Important!)

To train the model yourself, you must place the dataset provided by the hackathon organizers in the expected directory structure.

From the training script:

```python
data_dir = "../Offroad_Segmentation_Training_Dataset/train"
val_dir  = "../Offroad_Segmentation_Training_Dataset/val"
```

Your project structure must look like this:

```
your_project/
│
├── Offroad_Segmentation_Scripts/
│   ├── train.py
|   |── test.py
|
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   │
│   └── val/
│       ├── Color_Images/
│       └── Segmentation/
├── Offroad_Segmentation_testImages/
│   ├── Color_Images/
│   └── Segmentation/

```

Each image must have a corresponding mask with the same filename.

⚠️ If the dataset is not placed exactly in this structure, training will fail.

---

# 🚀 How to Train

```bash
python train.py
```


## 🧠 Architecture Overview

### 🔹 Backbone: DINOv2 (Frozen)

We use a frozen **DINOv2 ViT-S/14** backbone to extract multi-scale transformer features.

- 4 intermediate layers extracted
- Patch size: 14
- Token grid: `28 × 48`
- No fine-tuning (memory efficient & stable)

---

### 🔹 Pixel Decoder (Dual Resolution FPN)

The pixel decoder produces:

1. **Low-resolution FPN features** (28×48)  
   → Used for memory-efficient cross-attention  

2. **High-resolution features** (224×384)  
   → Used for sharp mask prediction  

Architecture components:

- Lateral 1×1 projections  
- Top-down FPN fusion  
- Progressive 3-stage transposed convolution upsampling  
- Residual refinement blocks  

---

### 🔹 Transformer Decoder

Mask2Former-style iterative masked decoding:

- 100 learnable object queries  
- 3 decoder layers  
- Multi-head cross-attention (low-res only)  
- Self-attention  
- Feedforward refinement  
- Dynamic attention masks per layer  

---

### 🔹 Mask Prediction Head

Each query predicts:

- Class logits (num_classes + background)
- Mask embeddings (dot-product with high-res features)

Final masks are refined by a lightweight CNN:

```
Conv → BN → GELU → Conv → BN → GELU → Conv
```

This improves boundary sharpness significantly.

---

## 🖼 Model Architecture Diagram

```
Input Image (392 × 672)
        │
        ▼
Frozen DINOv2 Backbone
        │
        ▼
Multi-scale Transformer Features (4 levels)
        │
        ▼
High-Res FPN Pixel Decoder
   ├── Low-res (28×48) → Cross-Attention Memory
   └── High-res (224×384) → Mask Prediction
        │
        ▼
Masked Transformer Decoder (3 layers)
        │
        ▼
Class + Mask Prediction
        │
        ▼
Mask Refinement CNN
        │
        ▼
Final Semantic Segmentation Map
```

---

# 📊 Test Set Results

**Mask2Former v3 — Dual-Resolution, Class-Balanced**

### 🔹 Overall Metrics

| Metric | Score |
|--------|--------|
| **Mean IoU** | **0.2805** |
| **Mean Dice** | **0.4743** |
| **Mean Pixel Accuracy** | **0.6422** |

---

### 🔹 Per-Class IoU

| Class | IoU |
|-------|------|
| Trees | 0.0075 |
| Lush Bushes | 0.0024 |
| Dry Grass | 0.4374 |
| Dry Bushes | 0.1617 |
| Ground Clutter | 0.0000 |
| Rocks | 0.0593 |
| Landscape | 0.6040 |
| Sky | 0.9582 |

---

### 🔹 Per-Class Dice

| Class | Dice |
|-------|------|
| Trees | 0.0281 |
| Lush Bushes | 0.0279 |
| Dry Grass | 0.6077 |
| Dry Bushes | 0.2445 |
| Ground Clutter | 0.0000 |
| Rocks | 0.1059 |
| Landscape | 0.7502 |
| Sky | 0.9786 |

---

## 📌 Observations

- Sky and Landscape are learned extremely well.
- Dry Grass segmentation is strong.
- Minority vegetation classes remain challenging despite class balancing.
- Ground Clutter and Lush Bushes remain underrepresented.

---

# ⚙️ Training Details

| Hyperparameter | Value |
|---------------|--------|
| Input Resolution | 392 × 672 |
| Batch Size | 2 |
| Epochs | 10 |
| Optimizer | AdamW |
| LR | 1e-4 |
| Weight Decay | 0.01 |
| Scheduler | Cosine Annealing |
| Queries | 100 |
| Decoder Layers | 3 |
| Hidden Dim | 256 |

---

# 🧮 Loss Function

Total Loss:

```
0.4 * Focal Cross Entropy
+ 0.4 * Weighted Dice
+ 0.2 * Boundary Loss
+ Deep Supervision (aux layers)
```

Class weights are computed using:

```
1 / log(1.02 + class_frequency)
```

Clamped and normalized for stability.

---

The script will:

- Compute class weights
- Load DINOv2 backbone
- Train segmentation head
- Save best model
- Save training curves
- Save full metric logs

Best model saved as:

```
potato2_segmentation_head_best.pth
```

Final model saved as:

```
potato2_segmentation_head_final.pth
```

---
