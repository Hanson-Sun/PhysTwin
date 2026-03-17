# Temporal Depth Smoothing: Post-Processing Layer
### A Depth-System-Agnostic Temporal Consistency Module

---

## 1. Problem Statement

Multi-camera depth systems like DA3 produce geometrically accurate, spatially consistent depth maps but suffer from **temporal jitter** — per-pixel depth values flicker between frames even when the underlying scene has not changed. This jitter arises from frame-independent inference: each frame is processed without any memory of previous frames.

The goal of this module is a **lightweight, post-processing network** that:
- Takes jittery depth maps from **any** depth system as input
- Outputs temporally smooth depth maps
- Preserves real object motion (does not over-smooth)
- Preserves geometric accuracy (does not drift from input values)
- Requires no retraining or modification of the upstream depth system

---

## 2. Design Principles

### System-Agnostic
The network must not assume:
- Any specific depth system (DA3, Marigold, DepthCrafter, stereo, etc.)
- Metric vs. affine-ambiguous depth
- Confidence maps or other auxiliary outputs
- Multi-camera vs. monocular input

Only two inputs are assumed to always be available: **depth maps** and **RGB frames**.

### Offline / Bidirectional
This is a post-processing module operating on recorded footage. It sees the full temporal window — past and future frames — which allows bidirectional temporal reasoning. This is significantly stronger than causal/online approaches and simpler to train.

### Residual Learning
The network predicts a **correction** Δ(t) per pixel, not the depth itself:

```
depth_smooth(t) = depth_raw(t) + Δ(t)
```

This ensures the network can never produce outputs wildly different from the input geometry — it can only make small adjustments. It also makes training easier since the network starts from a near-identity function.

---

## 3. Training Data Strategy

### The Core Idea
Use VDA (Video Depth Anything) as a **free source of temporal supervision**. VDA produces temporally smooth depth maps via learned temporal attention, but its absolute depth values are scale-ambiguous. DA3 produces geometrically correct, metric-anchored depth maps but is temporally jittery.

The insight: use VDA's **temporal structure** (how depth changes between frames) as the training signal, not VDA's absolute values.

### Data Generation Pipeline

```
Your multi-camera video footage
          │
          ├──► DA3 (multi-cam) ──► depth_raw [T, H, W]    (jittery, geometric)
          │
          └──► VDA (monocular, per camera) ──► depth_vda [T, H, W]  (smooth, wrong scale)
                          │
                          ▼
          Background mask (pixels where stddev(depth_raw) < θ over full clip)
                          │
                          ▼
          Per-clip scale/shift alignment:
            fit [s, b] via least squares on background pixels only
            depth_vda_aligned = s * depth_vda + b
                          │
                          ▼
          Sanity check: plot depth_raw vs depth_vda_aligned on background
          They should be near-identical after alignment ← CRITICAL VALIDATION STEP
```

### Why Background-Only Alignment Matters
The moving object violates the static scene assumption. If it is included in the scale/shift fit, VDA's temporally smooth object trajectory would be mapped to DA3's jittery object depth — corrupting the training signal entirely. The static background is the only reliable anchor.

### Training Diversity
Train on multiple depth systems to prevent the network from overfitting to DA3's specific failure modes:

| Depth System | Role |
|---|---|
| DA3 (multi-cam) | Primary input source |
| Marigold | Additional jittery input diversity |
| DepthCrafter | Additional jittery input diversity |
| VDA-aligned | Training target (temporal supervision) |

---

## 4. Network Architecture

### Overview

```
Input:
  depth_raw   [B, T, H, W]        ← any depth system, any scale
  rgb_frames  [B, T, H, W, 3]     ← corresponding RGB

  ↓ Per-clip normalisation
  depth_norm = (depth_raw - mean) / std   (per clip)

Processing:
  RGB encoder   → edge maps, motion features   [B, T, H, W, C_rgb]
  Depth encoder → depth features               [B, T, H, W, C_d]
  Concat        →                              [B, T, H, W, C]

  Temporal reasoning block (bidirectional, sees all T frames)

  Residual head → Δ_norm(t)   [B, T, H, W, 1]

Output:
  depth_smooth = (depth_norm + Δ_norm) * std + mean   ← denormalise
```

### Why Scale Normalisation Is Critical
Different depth systems produce values on completely different scales — DA3 metric (0.1m–10m), Marigold relative (0–1), stereo disparity (0–255). Normalising per-clip maps all inputs to the same statistical space, making the network truly system-agnostic. Denormalisation on the way out preserves the original scale.

### Recommended Architecture: 3D CNN (Start Here)

Simple, fast, no sequential dependencies. Processes all T frames in parallel.

```python
class TemporalDepthSmoother(nn.Module):
    def __init__(self, T=16, base_channels=32):
        super().__init__()

        # Lightweight per-frame feature extractors
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        )
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        )

        # 3D conv temporal reasoning (T, H, W)
        # kernel (3,3,3) = 1 frame temporal context each side
        # kernel (5,3,3) = 2 frames temporal context each side
        self.temporal_block = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3,3,3), padding=(1,1,1)), nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3,3,3), padding=(1,1,1)), nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=(5,3,3), padding=(2,1,1)), nn.ReLU(),
        )

        # Residual prediction head
        self.residual_head = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, depth_raw, rgb):
        B, T, H, W = depth_raw.shape

        # Per-clip normalisation
        mean = depth_raw.mean(dim=[1,2,3], keepdim=True)
        std  = depth_raw.std(dim=[1,2,3], keepdim=True).clamp(min=1e-6)
        depth_norm = (depth_raw - mean) / std

        # Per-frame feature extraction (apply 2D conv across batch*time)
        rgb_flat   = rgb.view(B*T, 3, H, W)
        depth_flat = depth_norm.view(B*T, 1, H, W)
        rgb_feat   = self.rgb_encoder(rgb_flat).view(B, T, -1, H, W)
        depth_feat = self.depth_encoder(depth_flat).view(B, T, -1, H, W)

        # Concat and reorder to [B, C, T, H, W] for Conv3D
        feat = torch.cat([rgb_feat, depth_feat], dim=2)
        feat = feat.permute(0, 2, 1, 3, 4)

        # Temporal reasoning
        feat = self.temporal_block(feat)

        # Residual
        delta_norm = self.residual_head(feat).squeeze(1)  # [B, T, H, W]
        delta_norm = delta_norm.permute(0, 2, 1, 3)       # reorder if needed

        # Reconstruct and denormalise
        depth_smooth_norm = depth_norm + delta_norm
        depth_smooth = depth_smooth_norm * std + mean

        return depth_smooth
```

**Approximate size:** ~500K parameters. Fast to train, fast at inference.

---

## 5. Alternative Architectures

### Option B: Temporal Transformer (More Expressive)

Replace the 3D conv block with a temporal attention mechanism. Each spatial position independently attends across all T timesteps.

```python
class TemporalAttentionBlock(nn.Module):
    """
    For each spatial position (h, w), apply multi-head self-attention
    across the T temporal dimension. Bidirectional — no causal mask.
    """
    def __init__(self, channels=32, num_heads=4, T=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.ff   = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(),
            nn.Linear(channels * 2, channels)
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        # Reshape: treat each spatial position as an independent sequence
        x = x.permute(0, 3, 4, 2, 1).reshape(B*H*W, T, C)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x + self.ff(x)
        x = x.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)
        return x
```

Use this if the 3D CNN baseline shows insufficient temporal coherence over long sequences. More expressive but ~3-5× slower.

### Option C: ConvLSTM (If Online Inference Is Needed Later)

Not needed for offline use, but included for reference if the module ever needs to be adapted for real-time:

```python
# ConvLSTM processes frames sequentially
# Hidden state carries temporal memory forward
# For offline use: run bidirectionally and average
#   forward pass:  t=0 → T
#   backward pass: t=T → 0
#   output: average of forward and backward hidden states
```

**Not recommended for this use case** — 3D CNN or temporal transformer will be superior for offline processing.

---

## 6. Loss Function

```python
def total_loss(depth_smooth, depth_raw, depth_vda_aligned, rgb, motion_mask,
               λ1=1.0, λ2=0.5, λ3=0.3):

    # ── L_temporal ──────────────────────────────────────────────────────────
    # How depth CHANGES between frames should match VDA's changes.
    # This is the core training signal. We supervise the derivative, not
    # the absolute value — this preserves real object motion.
    grad_smooth = depth_smooth[:, 1:] - depth_smooth[:, :-1]   # [B, T-1, H, W]
    grad_vda    = depth_vda_aligned[:, 1:] - depth_vda_aligned[:, :-1]
    L_temporal  = F.l1_loss(grad_smooth, grad_vda)

    # ── L_geometric ─────────────────────────────────────────────────────────
    # Stay close to the input on geometrically stable regions.
    # Edge-aware weight: relax at RGB boundaries (depth uncertain there),
    # enforce strongly on flat RGB regions (should be stable).
    rgb_grad = compute_image_gradient(rgb)                       # [B, T, H, W]
    w        = torch.exp(-5.0 * rgb_grad)                        # high at flat regions
    L_geometric = (torch.abs(depth_smooth - depth_raw) * w).mean()

    # ── L_smooth ─────────────────────────────────────────────────────────────
    # On static regions, depth should not change at all.
    # motion_mask: 1 = moving (relax), 0 = static (enforce)
    L_smooth = (torch.abs(grad_smooth) * (1 - motion_mask[:, 1:])).mean()

    return λ1 * L_temporal + λ2 * L_geometric + λ3 * L_smooth


def compute_image_gradient(rgb):
    # Sobel gradient magnitude across spatial dimensions, averaged over channels
    # rgb: [B, T, H, W, 3]
    ...

def compute_motion_mask(rgb, threshold=0.05):
    # Frame differencing: pixels that change significantly are "moving"
    diff = torch.abs(rgb[:, 1:] - rgb[:, :-1]).mean(dim=-1)
    return (diff > threshold).float()
```

### Loss Term Summary

| Term | Purpose | What it enforces |
|---|---|---|
| `L_temporal` | Core training signal from VDA | Temporal *change* matches VDA's smooth trajectory |
| `L_geometric` | Preserve upstream geometry | Output stays close to input on stable regions |
| `L_smooth` | Suppress noise on background | Static pixels should have zero depth change |

---

## 7. Inference: Sliding Window

For long videos, use a sliding window with overlap and blend:

```python
def sliding_window_inference(model, depth_raw, rgb, T=16, stride=8):
    N = depth_raw.shape[0]  # total frames
    output = torch.zeros_like(depth_raw)
    weights = torch.zeros(N)

    # Raised cosine window for smooth blending at boundaries
    window = raised_cosine_window(T)

    for start in range(0, N - T + 1, stride):
        end = start + T
        chunk_depth = depth_raw[start:end]
        chunk_rgb   = rgb[start:end]

        smoothed = model(chunk_depth.unsqueeze(0), chunk_rgb.unsqueeze(0))
        smoothed = smoothed.squeeze(0)

        output[start:end]  += smoothed * window
        weights[start:end] += window

    return output / weights.clamp(min=1e-6)
```

The raised cosine window ensures frames near chunk boundaries (which have less temporal context) contribute less to the final output — reducing boundary artifacts.

---

## 8. Implementation Roadmap

### Week 1 — Data Pipeline
- [ ] Run DA3 on all footage, save depth maps + timestamps
- [ ] Run VDA on same footage (per camera, monocular), save depth maps
- [ ] Implement background mask: `stddev(depth_raw) < θ` over full clip
- [ ] Implement per-clip scale/shift alignment (background only)
- [ ] **Sanity check**: plot `depth_raw` vs `depth_vda_aligned` on background pixels over time — they must be near-identical before proceeding

### Week 2 — Network Implementation
- [ ] Implement `TemporalDepthSmoother` (3D CNN variant)
- [ ] Implement per-clip normalisation/denormalisation wrapper
- [ ] Implement loss functions (`L_temporal`, `L_geometric`, `L_smooth`)
- [ ] Implement motion mask from RGB frame differencing
- [ ] Unit test: does output shape match input? Does Δ=0 pass through correctly?

### Week 3 — Training
- [ ] Train on small subset (5–10 clips) — verify all losses decrease
- [ ] Scale to full dataset
- [ ] Log: temporal variance on static background pixels (should drop)
- [ ] Log: mean absolute deviation from `depth_raw` on background (should stay low)
- [ ] Checkpoint every epoch

### Week 4 — Evaluation & Tuning
- [ ] Visual: does background flicker stop?
- [ ] Visual: does object motion remain sharp and natural?
- [ ] Quantitative: temporal variance on static pixels vs DA3 baseline
- [ ] Quantitative: depth error on background vs DA3 baseline
- [ ] Test on non-DA3 depth input (Marigold or DepthCrafter) to verify system-agnostic claim
- [ ] Tune loss weights if needed (increase λ2 if network drifts from geometry)

---

## 9. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| VDA alignment is bad → corrupted training signal | Validate visually before training. If background diverges, increase mask threshold θ |
| Network learns to copy VDA (ignores input geometry) | Monitor L_geometric. Increase λ2 if depth drifts from input |
| Network over-smooths real object motion | Tune motion mask threshold. Verify L_smooth is only applied on static regions |
| Poor generalisation to other depth systems | Train on Marigold + DepthCrafter alongside DA3 |
| Sliding window boundary artifacts | Use raised cosine blending window |
| Scale normalisation fails on near-flat depth clips | Clamp std with min value (e.g. 1e-6) to prevent division by zero |

---

## 10. Related Literature

This module sits at the intersection of three research areas:

### Blind Video Temporal Consistency (Most Relevant)

**Blind Video Temporal Consistency** (Bonneel et al., SIGGRAPH 2015) — the foundational paper. Proposes using the *original unprocessed video* as a temporal guide to stabilise a processed version. Exactly the paradigm we are following: use the raw RGB to guide stabilisation of the processed depth.

**Learning Blind Video Temporal Consistency** (Lai et al., ECCV 2018) — first deep learning approach to blind temporal consistency. Uses a deep recurrent network that takes the original and per-frame processed video as inputs to learn temporal consistency. *This is the closest paper to what we are building.* Key difference from our approach: they use a recurrent (online) architecture; we use bidirectional 3D conv/attention for offline processing, which is stronger.

**Blind Video Temporal Consistency via Deep Video Prior** (Lei et al., NeurIPS 2020) — trains a network on a single video pair (original + processed) without a large dataset. Useful as a test-time fine-tuning strategy if generalisation is insufficient.

**Towards Practical Consistent Video Depth Estimation** (ICMR 2023) — the most directly relevant recent paper. A plug-and-play post-processing method using a deep recurrent network specifically for depth maps. Takes adjacent original and optimised depth maps as inputs. *System-agnostic and applied to multiple depth estimation models* — exactly the design goal here. Key limitation: online/recurrent, not bidirectional.

### Temporal Depth Estimation (Contextual)

**Video Depth Anything (VDA)** (2024) — our source of training supervision. Uses a spatiotemporal head with temporal self-attention and a temporal gradient matching loss. The gradient matching loss is what we are adopting as `L_temporal`.

**Online VDA (oVDA)** (2025) — caches latent features for online inference. Relevant if this module ever needs to be adapted for real-time use.

**Enforcing Temporal Consistency in Video Depth Estimation** (Li et al., ICCVW 2021) — explicit temporal consistency loss for depth, applied during training. Informs our `L_temporal` design.

**FutureDepth** (ECCV 2024) — uses future frame prediction as an auxiliary task to enforce temporal consistency. The future-prediction auxiliary loss is an interesting addition to consider if `L_temporal` alone is insufficient.

**Temporally Consistent Depth Estimation with Recurrent Architectures** (Tananaev et al., ECCV 2018) — ConvLSTM-based temporal depth estimation. Demonstrates that smoothing the temporal *state* (not just the output) avoids negative effects on dynamic scenes — validates our motion-aware approach.

### Depth Refinement Networks (Architectural Inspiration)

**ChronoDepth** (2024) — uses video diffusion priors for temporally consistent depth. Proposes a consistent context-aware training strategy with sliding window and overlapping frame initialisation. Directly informs our sliding window inference design.

**IterDepth** — iterative residual refinement for depth estimation. Validates the residual learning paradigm for depth: predict Δ rather than absolute depth.

---

## 11. Summary

```
Any depth system (DA3, Marigold, stereo, etc.)
              ↓
         depth_raw [T, H, W]   +   rgb [T, H, W, 3]
              ↓
    ┌─────────────────────────┐
    │  Per-clip normalisation  │
    │  RGB encoder             │
    │  Depth encoder           │
    │  3D Conv temporal block  │  ← sees past + future (offline)
    │  Residual head → Δ(t)   │
    │  Denormalisation         │
    └─────────────────────────┘
              ↓
      depth_smooth [T, H, W]
      Geometrically faithful + temporally consistent
```

**Training signal:** VDA-aligned temporal gradients  
**Geometric anchor:** Edge-aware fidelity to input  
**Motion preservation:** RGB motion mask on `L_smooth`  
**Scale invariance:** Per-clip normalisation/denormalisation  
**Model size:** ~500K parameters (3D CNN), ~2M (temporal transformer)
