"""Pure model signal inspection — no demo/FAISS code.

Loads one valid sample, runs forward pass with hooks on every layer,
and prints exactly which signals have variance worth visualizing.

Run:  cd /media/urlab/KINGSTON/aic && python -m source.demo.test_inspect_signals
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "source"
for p in [str(SRC), str(SRC / "train")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from demo.config import BASE_DIR, CHECKPOINT, MODEL_CFG, DATA_CFG
from model import build_fusion_encoder, FusionConfig
from model.attention import GatedAttentionVideoRoPE
from model.blocks import FusionBlockLocalGlobal
from model.components import SwiGLUFFN
from dataset import VietnameseNewsDataset, collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════
# 1. Load model + one valid sample
# ═══════════════════════════════════════════════════════════════════════
print("Loading model …")
model = build_fusion_encoder(FusionConfig(**MODEL_CFG)).to(DEVICE)
model.load_state_dict(torch.load(str(CHECKPOINT), map_location=DEVICE))
model.eval()

print("Finding a sample with non-zero visual features …")
ds = VietnameseNewsDataset("test", BASE_DIR,
    max_frames=DATA_CFG["max_frames"],
    max_segments=DATA_CFG["max_segments"],
    max_seg_tokens=DATA_CFG["max_seg_tokens"],
    visual_feature_subdir=DATA_CFG["visual_feature_subdir"])

vis_dir = BASE_DIR / "data" / "features" / DATA_CFG["visual_feature_subdir"]
sample = None
for s in ds.samples:
    npz = vis_dir / s["video_id"] / f"{s['video_id']}_{s['shot_id']}.npz"
    if npz.exists() and float(np.load(npz)["features"].max()) > 0:
        sample = ds[ds.samples.index(s)]
        break
assert sample is not None, "No valid sample found"
batch = collate_fn([sample])
sid = batch["shot_id"][0]
n_frames = int(batch["visual_mask"].sum().item())
n_segs = int(batch["segment_mask"].sum().item())
print(f"Sample: {sid}  |  frames={n_frames}  segs={n_segs}  "
      f"vis_norm={batch['visual_features'].norm():.1f}\n")

# ═══════════════════════════════════════════════════════════════════════
# 2. Instrument model — hooks on everything
# ═══════════════════════════════════════════════════════════════════════
H = {}   # collected tensors
hooks = []

def hook(name):
    def fn(_, inp, out):
        t = out if torch.is_tensor(out) else (out[0] if isinstance(out, tuple) else None)
        if t is not None:
            H[name] = t.detach().float().cpu()
    return fn

# Enable attention capture
for m in model.modules():
    if isinstance(m, GatedAttentionVideoRoPE):
        m._capture_attn = True

for bi, blk in enumerate(model.blocks):
    if not isinstance(blk, FusionBlockLocalGlobal):
        continue
    B = f"B{bi}"
    # attention modules
    for aname in ("self_attn", "local_ca", "global_ca"):
        a = getattr(blk, aname)
        hooks.append(a.register_forward_hook(hook(f"{B}.{aname}_out")))
        hooks.append(a.w_q.register_forward_hook(hook(f"{B}.{aname}.Wq")))
        hooks.append(a.w_k.register_forward_hook(hook(f"{B}.{aname}.Wk")))
        hooks.append(a.w_v.register_forward_hook(hook(f"{B}.{aname}.Wv")))
        hooks.append(a.w_o.register_forward_hook(hook(f"{B}.{aname}.Wo")))
    # FFN / norms
    hooks.append(blk.ffn1.register_forward_hook(hook(f"{B}.ffn1")))
    hooks.append(blk.ffn2.register_forward_hook(hook(f"{B}.ffn2")))
    hooks.append(blk.norm1.register_forward_hook(hook(f"{B}.norm1")))
    hooks.append(blk.norm2.register_forward_hook(hook(f"{B}.norm2")))
    hooks.append(blk.norm3.register_forward_hook(hook(f"{B}.norm3")))
    hooks.append(blk.norm4.register_forward_hook(hook(f"{B}.norm4")))

hooks.append(model.visual_proj.register_forward_hook(hook("visual_proj")))
hooks.append(model.final_norm.register_forward_hook(hook("final_norm")))

# ═══════════════════════════════════════════════════════════════════════
# 3. Forward
# ═══════════════════════════════════════════════════════════════════════
with torch.inference_mode(), torch.amp.autocast(
        "cuda" if DEVICE.type == "cuda" else "cpu",
        enabled=DEVICE.type == "cuda", dtype=torch.bfloat16):
    e_plus, e_narr, temperature = model(
        seg_tokens=batch["token_reprs"].to(DEVICE),
        seg_pooled=batch["segment_pooled"].to(DEVICE),
        visual_features=batch["visual_features"].to(DEVICE),
        seg_timestamps=batch["seg_timestamps"].to(DEVICE),
        frame_timestamps=batch["frame_timestamps"].to(DEVICE),
        seg_mask=batch["segment_mask"].to(DEVICE),
        visual_mask=batch["visual_mask"].to(DEVICE),
        query_emb=None,
        token_mask=batch["token_mask"].to(DEVICE))

# Collect attention weights
for bi, blk in enumerate(model.blocks):
    if not isinstance(blk, FusionBlockLocalGlobal):
        continue
    for aname in ("self_attn", "local_ca", "global_ca"):
        a = getattr(blk, aname)
        if a._last_attn is not None:
            H[f"B{bi}.{aname}.softmax"] = a._last_attn[0].float().cpu()

# Cleanup
for h in hooks:
    h.remove()
for m in model.modules():
    if isinstance(m, GatedAttentionVideoRoPE):
        m._capture_attn = False
        m._last_attn = None

# ═══════════════════════════════════════════════════════════════════════
# 4. Report: Gates
# ═══════════════════════════════════════════════════════════════════════
def sec(title):
    print(f"\n{'═'*80}\n  {title}\n{'═'*80}")

sec("A. SIGMOID GATE VALUES")
for bi, blk in enumerate(model.blocks):
    if not isinstance(blk, FusionBlockLocalGlobal):
        continue
    for aname in ("self_attn", "local_ca", "global_ca"):
        g = torch.sigmoid(getattr(blk, aname).gate).detach().cpu()
        print(f"  B{bi}.{aname:12s}  sigmoid ∈ [{g.min():.4f}, {g.max():.4f}]  mean={g.mean():.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 5. Report: All intermediate norms
# ═══════════════════════════════════════════════════════════════════════
sec("B. ALL INTERMEDIATE TENSOR NORMS")
print(f"  {'name':42s} {'shape':22s} {'norm':>10s} {'min':>10s} {'max':>10s} {'std':>10s}")
print(f"  {'─'*42} {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
for k in sorted(H.keys()):
    if "softmax" in k:
        continue  # printed separately
    t = H[k]
    sh = str(list(t.shape))
    print(f"  {k:42s} {sh:22s} {t.norm().item():10.2f} {t.min().item():10.4f} "
          f"{t.max().item():10.4f} {t.std().item():10.6f}")

# ═══════════════════════════════════════════════════════════════════════
# 6. Report: Attention weight analysis
# ═══════════════════════════════════════════════════════════════════════
def ent(w):
    w = w.flatten(); w = w[w > 0]
    return float(-np.sum(w * np.log2(w))) if len(w) > 0 else 0.0

sec("C. ATTENTION WEIGHT ANALYSIS (post-softmax)")
for bi in range(len(model.blocks)):
    for aname in ("self_attn", "local_ca", "global_ca"):
        key = f"B{bi}.{aname}.softmax"
        if key not in H:
            continue
        p = H[key]  # [H, Tq, Tk]
        nH, Tq, Tk = p.shape
        max_e = np.log2(Tk)
        rv = p.var(dim=-1)
        rm = p.max(dim=-1).values

        print(f"\n  {key}  [{nH}×{Tq}×{Tk}]")
        print(f"    row_var: mean={rv.mean():.8f} max={rv.max():.8f}")
        print(f"    row_max: mean={rm.mean():.4f}  max={rm.max():.6f}")
        print(f"    max_entropy (uniform) = {max_e:.2f} bits")
        # per-head
        for hi in range(nH):
            rows = p[hi, :min(Tq, 30)].numpy()
            es = [ent(r / max(r.sum(), 1e-12)) for r in rows if r.sum() > 0]
            me = np.mean(es) if es else 0
            marker = "🔥 FOCUSED" if me < max_e * 0.85 else ("⚡ partial" if me < max_e * 0.95 else "⬜ uniform")
            print(f"    H{hi:02d}: avg_row_ent={me:5.2f}/{max_e:.2f}  {marker}")

# ═══════════════════════════════════════════════════════════════════════
# 7. Report: Residual contribution
# ═══════════════════════════════════════════════════════════════════════
sec("D. RESIDUAL STREAM CONTRIBUTION (L2 norm)")
for bi in range(len(model.blocks)):
    parts = []
    for label in ("self_attn_out", "local_ca_out", "global_ca_out", "ffn1", "ffn2"):
        k = f"B{bi}.{label}"
        if k in H:
            parts.append((label, H[k].norm().item()))
    total = sum(v for _, v in parts)
    print(f"\n  Block {bi}:")
    for label, norm in parts:
        pct = norm / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {label:18s}  norm={norm:8.1f}  ({pct:5.1f}%) {bar}")

# ═══════════════════════════════════════════════════════════════════════
# 8. Report: Cross-modal alignment
# ═══════════════════════════════════════════════════════════════════════
sec("E. CROSS-MODAL COSINE SIMILARITY (projected visual ↔ text tokens)")
vis = H.get("visual_proj")
if vis is not None and vis.norm() > 0:
    vf = vis[0, :n_frames].reshape(-1, vis.shape[-1])           # [N*16, D]
    tf = batch["token_reprs"][0].reshape(-1, 1024)               # [M*T, D]
    tm = batch["token_mask"][0].reshape(-1) > 0
    tf = tf[tm]                                                   # [valid, D]

    vn = F.normalize(vf.float(), dim=-1)
    tn = F.normalize(tf.float(), dim=-1)
    sim = tn @ vn.T                                               # [text, visual]

    print(f"  Shape: {list(sim.shape)}  range=[{sim.min():.4f}, {sim.max():.4f}]  "
          f"mean={sim.mean():.4f}  std={sim.std():.4f}")

    # per-frame
    print(f"\n  Per-frame avg cosine (text → visual):")
    for fi in range(min(n_frames, 15)):
        fs = sim[:, fi*16:(fi+1)*16]
        print(f"    F{fi:02d}: mean={fs.mean():.4f}  max={fs.max():.4f}  std={fs.std():.4f}")
else:
    print("  ⚠ visual_proj output is zero")

# ═══════════════════════════════════════════════════════════════════════
# 9. FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════
sec("F. VERDICT — WHAT IS WORTH SHOWING ON DEMO")

signals = []

# Check each attention type
for bi in range(len(model.blocks)):
    for aname in ("local_ca", "global_ca", "self_attn"):
        key = f"B{bi}.{aname}.softmax"
        out_key = f"B{bi}.{aname}_out"
        if key not in H:
            continue
        p = H[key]
        nH, Tq, Tk = p.shape
        max_e = np.log2(Tk)
        # mean row entropy across all heads
        all_ent = []
        for hi in range(nH):
            rows = p[hi, :min(Tq, 50)].numpy()
            es = [ent(r / max(r.sum(), 1e-12)) for r in rows if r.sum() > 0]
            if es:
                all_ent.extend(es)
        avg_ent = np.mean(all_ent) if all_ent else max_e
        ent_ratio = avg_ent / max_e

        out_norm = H[out_key].norm().item() if out_key in H else 0
        alive = out_norm > 1.0
        focused = ent_ratio < 0.85

        status = "✅ SHOW" if (alive and focused) else ("⚠️  weak" if alive else "❌ dead")
        signals.append({
            "name": f"B{bi}.{aname}", "status": status,
            "ent_ratio": ent_ratio, "out_norm": out_norm,
            "focused": focused, "alive": alive
        })
        print(f"  {status:10s}  B{bi}.{aname:12s}  "
              f"ent={avg_ent:.2f}/{max_e:.2f} ({ent_ratio:.0%})  "
              f"out_norm={out_norm:.1f}")

# Cross-modal sim
if vis is not None and vis.norm() > 0:
    print(f"  {'✅ SHOW':10s}  cross_modal_cosine_sim   "
          f"range=[{sim.min():.3f},{sim.max():.3f}] std={sim.std():.4f}")
else:
    print(f"  {'❌ dead':10s}  cross_modal_cosine_sim   visual_proj=0")

# FFN
for bi in range(len(model.blocks)):
    for fn in ("ffn1", "ffn2"):
        k = f"B{bi}.{fn}"
        if k in H:
            n = H[k].norm().item()
            print(f"  {'✅ SHOW' if n > 10 else '❌ dead':10s}  B{bi}.{fn:18s}  norm={n:.1f}")

print(f"\n  Temperature = {torch.exp(temperature).item():.4f}")
print(f"  e_plus norm = {e_plus.norm().item():.4f}")
print(f"  e_narr norm = {e_narr.norm().item():.4f}")
print("\nDone.")
