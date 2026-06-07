"""Training & evaluation utilities shared by every ablation variant.

Extracted from ``notebooks/ablation_study_1.ipynb`` so the notebook stays
focused on **what** is being ablated, while the helpers ( query encoding,
losses, retrieval evaluation ) live in one reusable module.

All functions are dependency-injected: the caller passes the BGE-M3
tokenizer/model, the device, and the data paths — no hidden globals.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Query encoding
# ---------------------------------------------------------------------------

def encode_queries(texts: List[str], tokenizer, encoder_model,
                   device: torch.device, max_length: int = 512,
                   batch_size: Optional[int] = None) -> torch.Tensor:
    """Mean-pool BGE-M3 last hidden state, L2-normalize. Returns ``[N, D]``."""
    if not texts:
        return torch.empty((0, encoder_model.config.hidden_size), device=device)
    if batch_size is None:
        batch_size = len(texts)
    all_embs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tokenizer(chunk, padding=True, truncation=True,
                            max_length=max_length, return_tensors='pt').to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    h = encoder_model(**enc, return_dict=True).last_hidden_state
            else:
                h = encoder_model(**enc, return_dict=True).last_hidden_state
            m = enc['attention_mask'].unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
            all_embs.append(F.normalize(pooled.float(), p=2, dim=-1))
    return torch.cat(all_embs, dim=0)


def encode_hard_negs(hard_neg_texts_batch: List[List[str]], tokenizer,
                     encoder_model, device: torch.device,
                     max_length: int = 512) -> Optional[torch.Tensor]:
    """Pack ragged hard-negative lists into a ``[B, K, D]`` tensor (pad with ``''``)."""
    b = len(hard_neg_texts_batch)
    k = max((len(x) for x in hard_neg_texts_batch), default=0)
    if k == 0:
        return None
    flat = []
    for negs in hard_neg_texts_batch:
        flat.extend(negs + [''] * (k - len(negs)))
    emb = encode_queries(flat, tokenizer, encoder_model, device, max_length=max_length)
    return emb.view(b, k, -1)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def symmetric_infonce(q: torch.Tensor, d: torch.Tensor,
                      temperature: torch.Tensor) -> torch.Tensor:
    tau = torch.exp(temperature).clamp_min(1e-6)
    sim = torch.matmul(q, d.T) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


def retrieval_loss(q_hat: torch.Tensor, e_plus: torch.Tensor, e_narr: torch.Tensor,
                   temperature: torch.Tensor,
                   hard_neg_embs: Optional[torch.Tensor] = None,
                   lambda_narr: float = 0.5, beta_hard: float = 0.3,
                   gamma_mrl: float = 0.1) -> torch.Tensor:
    """L = L_base + λ·L_narr + β·L_hard + γ·L_MRL.

    - ``lambda_narr=0`` ablates NarrLoss.
    - ``gamma_mrl=0``   ablates MRL.
    """
    l_base = symmetric_infonce(q_hat, e_plus, temperature)
    l_narr = torch.tensor(0.0, device=q_hat.device)
    if lambda_narr > 0:
        l_narr = symmetric_infonce(q_hat, e_narr, temperature)
    l_hard = torch.tensor(0.0, device=q_hat.device)
    if hard_neg_embs is not None and beta_hard > 0:
        q = q_hat.unsqueeze(1)
        neg_scores = torch.sum(q * hard_neg_embs, dim=-1)
        pos_scores = torch.sum(q_hat * e_plus, dim=-1, keepdim=True)
        l_hard = torch.relu(neg_scores - pos_scores + 0.1).mean()
    l_mrl = torch.tensor(0.0, device=q_hat.device)
    if gamma_mrl > 0:
        for d_dim in [64, 128, 256, 512, 1024]:
            l_mrl = l_mrl + symmetric_infonce(q_hat[:, :d_dim], e_plus[:, :d_dim], temperature)
        l_mrl = l_mrl / 5.0
    return l_base + lambda_narr * l_narr + beta_hard * l_hard + gamma_mrl * l_mrl


# ---------------------------------------------------------------------------
# Document / query loading + retrieval evaluation
# ---------------------------------------------------------------------------

def precompute_docs(model: torch.nn.Module, loader, device: torch.device):
    """Run encoder over a split loader; returns ``(normalized_embs, shot_ids)``."""
    model.eval()
    doc_embs, shot_ids = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc='Docs', leave=False, dynamic_ncols=True):
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu',
                                   enabled=(device.type == 'cuda'), dtype=torch.bfloat16):
                e_plus, _, _ = model(
                    seg_tokens=batch['token_reprs'].to(device),
                    seg_pooled=batch['segment_pooled'].to(device),
                    visual_features=batch['visual_features'].to(device),
                    seg_timestamps=batch['seg_timestamps'].to(device),
                    frame_timestamps=batch['frame_timestamps'].to(device),
                    seg_mask=batch['segment_mask'].to(device),
                    visual_mask=batch['visual_mask'].to(device),
                    query_emb=None,
                    token_mask=batch['token_mask'].to(device),
                )
            doc_embs.append(e_plus.detach().cpu())
            shot_ids.extend(batch['shot_id'])
    return F.normalize(torch.cat(doc_embs, dim=0), p=2, dim=-1), shot_ids


def find_query_dir(base_dir: Path, split: str) -> Path:
    for d in [base_dir / 'data' / split, base_dir / 'data' / f'{split}2']:
        if d.exists() and any(d.glob('*.json')):
            return d
    raise FileNotFoundError(f'No query JSON found for split={split} under {base_dir}')


def load_queries(base_dir: Path, split: str) -> List[dict]:
    """Load and dedupe ``{shot_id, query}`` items from per-video JSON files."""
    q_items = []
    for jf in sorted(find_query_dir(base_dir, split).glob('*.json')):
        video_id = jf.stem
        for row in json.loads(jf.read_text('utf-8')):
            sid = str(row.get('id', '')).zfill(3)
            q = str(row.get('positive', '')).strip()
            if sid and q:
                q_items.append({'shot_id': f'{video_id}_{sid}', 'query': q})
    seen, deduped = set(), []
    for x in q_items:
        if x['shot_id'] not in seen:
            seen.add(x['shot_id'])
            deduped.append(x)
    return deduped


def evaluate_retrieval(model: torch.nn.Module, loader, split: str,
                       device: torch.device, base_dir: Path,
                       tokenizer, encoder_model,
                       dual_softmax_tau: float = 0.01,
                       eval_query_max_length: int = 256,
                       eval_query_batch_size: int = 16) -> dict:
    """Compute Recall@{1,5,10}, MedR, MeanR using Dual-Softmax similarity."""
    doc_embs, shot_ids = precompute_docs(model, loader, device)
    q_items = load_queries(base_dir, split)
    q_embs_all = encode_queries(
        [x['query'] for x in q_items],
        tokenizer, encoder_model, device,
        max_length=eval_query_max_length,
        batch_size=eval_query_batch_size,
    )
    shot_to_idx = {s: i for i, s in enumerate(shot_ids)}
    filtered = [x for x in q_items if x['shot_id'] in shot_to_idx]
    q_idx = [q_items.index(x) for x in filtered]
    q_embs_f = q_embs_all[q_idx]
    gt_indices = [shot_to_idx[x['shot_id']] for x in filtered]
    sim_raw = torch.matmul(q_embs_f, doc_embs.to(device).T) / dual_softmax_tau
    sim_dsl = torch.softmax(sim_raw, dim=1) * torch.softmax(sim_raw, dim=0)
    sim_np = sim_dsl.detach().cpu().numpy()
    ranks = np.array([int(np.where(np.argsort(-sim_np[i]) == gt)[0][0]) + 1
                      for i, gt in enumerate(gt_indices)])
    return {
        'R1':   float((ranks <= 1).mean() * 100),
        'R5':   float((ranks <= 5).mean() * 100),
        'R10':  float((ranks <= 10).mean() * 100),
        'SumR': float(((ranks <= 1).mean() + (ranks <= 5).mean() + (ranks <= 10).mean()) * 100),
        'MedR': float(np.median(ranks)),
        'MeanR': float(ranks.mean()),
        'n_queries': len(ranks),
    }


__all__ = [
    "encode_queries", "encode_hard_negs",
    "symmetric_infonce", "retrieval_loss",
    "precompute_docs", "find_query_dir", "load_queries",
    "evaluate_retrieval",
]
