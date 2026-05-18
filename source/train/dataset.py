"""Dataset for D1: load precomputed visual features and BGE-M3 features."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VietnameseNewsDataset(Dataset):
    """Dataset for D1: load precomputed visual features and BGE-M3 features."""

    def __init__(
        self,
        split: str,
        base_dir: str | Path,
        max_frames: int = 18,
        max_segments: int = 10,
        max_seg_tokens: int = 128,
        visual_feature_subdir: str = "siglip2",
    ):
        if split not in ("train", "val", "test", "test_dedicated"):
            raise ValueError("Unsupported split: " + split)
        self.split = split
        self.base_dir = Path(base_dir)
        self.max_frames = max_frames
        self.max_segments = max_segments
        self.max_seg_tokens = max_seg_tokens
        self.visual_feature_subdir = visual_feature_subdir

        self.data_dir = self.base_dir / "data" / split
        self.visual_dir = self.base_dir / "data" / "features" / visual_feature_subdir
        self.bgem3_dir = self.base_dir / "data" / "features" / "bgem3"
        self.time_extract_dir = self.base_dir / "dataset" / "time_extract"
        self.transcripts_dir = self.base_dir / "dataset" / "transcripts"

        if not self.data_dir.exists():
            raise FileNotFoundError("Split dir not found: " + str(self.data_dir))

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError("No valid samples for split=" + split)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _norm_shot_id(shot_id: str) -> str:
        return str(shot_id).zfill(3)

    def _build_samples(self) -> List[Dict[str, object]]:
        grouped = {}
        for jf in sorted(self.data_dir.glob("*.json")):
            video_id = jf.stem
            rows = json.loads(jf.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sid = row.get("id", "")
                q = str(row.get("positive", "")).strip()
                if not sid or not q:
                    continue
                shot_id = self._norm_shot_id(sid)
                key = f"{video_id}_{shot_id}"
                if key not in grouped:
                    grouped[key] = {
                        "video_id": video_id,
                        "shot_id": shot_id,
                        "queries": [],
                        "hard_negatives": [],
                    }
                grouped[key]["queries"].append(q)
                grouped[key]["hard_negatives"].extend(
                    [str(x).strip() for x in row.get("negatives", []) if str(x).strip()]
                )

        samples = []
        for key, item in grouped.items():
            video_id = item["video_id"]
            shot_id = item["shot_id"]
            vis_path = self.visual_dir / video_id / f"{video_id}_{shot_id}.npz"
            tok_path = self.bgem3_dir / video_id / f"{shot_id}_tokens.npz"
            pool_path = self.bgem3_dir / video_id / f"{shot_id}_pooled.npz"
            frame_json = self.time_extract_dir / video_id / f"{video_id}_{shot_id}.json"
            seg_json = self.transcripts_dir / video_id / f"{shot_id}.json"
            if vis_path.exists() and tok_path.exists() and pool_path.exists():
                samples.append(item)

        if self.split == "train":
            random.shuffle(samples)
        return samples

    def _load_visual(self, video_id: str, shot_id: str):
        path = self.visual_dir / video_id / f"{video_id}_{shot_id}.npz"
        data = np.load(path)
        feat = torch.from_numpy(data["features"]).float()  # [N, 16, 1152]
        n = feat.shape[0]

        if n > self.max_frames:
            idx = np.linspace(0, n - 1, self.max_frames, dtype=int)
            feat = feat[idx]
            n = self.max_frames

        if n < self.max_frames:
            pad = torch.zeros(self.max_frames - n, 16, feat.shape[-1], dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=0)

        mask = torch.zeros(self.max_frames, dtype=torch.float32)
        mask[:min(n, self.max_frames)] = 1.0
        return feat, mask

    def _load_text(self, video_id: str, shot_id: str):
        tok_path = self.bgem3_dir / video_id / f"{shot_id}_tokens.npz"
        pool_path = self.bgem3_dir / video_id / f"{shot_id}_pooled.npz"

        pool_data = np.load(pool_path)
        token_keys = sorted(
            np.load(tok_path).files,
            key=lambda x: int(x.split("_")[-1]),
        )
        seg_arrays = [torch.from_numpy(np.load(tok_path)[k]).float() for k in token_keys]
        pooled = torch.from_numpy(pool_data["pooled"]).float()  # [M, 1024]

        m = min(len(seg_arrays), self.max_segments)

        # token_reprs: [max_segments, max_seg_tokens, 1024]
        token_reprs = torch.zeros(self.max_segments, self.max_seg_tokens, pooled.shape[-1],
                                  dtype=torch.float32)
        token_mask = torch.zeros(self.max_segments, self.max_seg_tokens, dtype=torch.float32)

        # segment_pooled: [max_segments, 1024]
        segment_pooled = torch.zeros(self.max_segments, pooled.shape[-1], dtype=torch.float32)
        segment_mask = torch.zeros(self.max_segments, dtype=torch.float32)

        use_m = min(m, self.max_segments)
        for i in range(use_m):
            arr = seg_arrays[i]
            ti = min(arr.shape[0], self.max_seg_tokens)
            token_reprs[i, :ti] = arr[:ti]
            token_mask[i, :ti] = 1.0
            segment_pooled[i] = pooled[i]
            segment_mask[i] = 1.0

        return token_reprs, token_mask, segment_pooled, segment_mask

    def _load_timestamps(
        self,
        video_id: str,
        shot_id: str,
        n_frames_raw: int,
        n_segments_raw: int,
    ):
        frame_json = self.time_extract_dir / video_id / f"{video_id}_{shot_id}.json"
        seg_json = self.transcripts_dir / video_id / f"{shot_id}.json"

        f_entries = json.loads(frame_json.read_text(encoding="utf-8"))
        s_entries = json.loads(seg_json.read_text(encoding="utf-8"))

        frame_ts = torch.tensor(
            [float(e["start"]) for e in f_entries],
            dtype=torch.float32,
        )
        seg_ts = torch.tensor(
            [float(s["start"]) for s in s_entries],
            dtype=torch.float32,
        )

        # Align frame_ts to max_frames (same linspace as visual)
        if frame_ts.numel() > self.max_frames:
            idx = np.linspace(0, frame_ts.numel() - 1, self.max_frames, dtype=int)
            frame_ts = frame_ts[idx]
        if frame_ts.numel() < self.max_frames:
            frame_ts = torch.cat([
                frame_ts,
                torch.zeros(self.max_frames - frame_ts.numel(), dtype=torch.float32),
            ], dim=0)

        if seg_ts.numel() > self.max_segments:
            seg_ts = seg_ts[:self.max_segments]
        if seg_ts.numel() < self.max_segments:
            seg_ts = torch.cat([
                seg_ts,
                torch.zeros(self.max_segments - seg_ts.numel(), dtype=torch.float32),
            ], dim=0)

        return frame_ts, seg_ts

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        shot_id = sample["shot_id"]

        visual_features, visual_mask = self._load_visual(video_id, shot_id)
        token_reprs, token_mask, segment_pooled, segment_mask = self._load_text(
            video_id, shot_id
        )
        n_frames_raw = int(visual_mask.sum().item())
        n_segments_raw = int(segment_mask.sum().item())
        frame_timestamps, seg_timestamps = self._load_timestamps(
            video_id, shot_id,
            n_frames_raw=n_frames_raw,
            n_segments_raw=n_segments_raw,
        )

        # For training: pick one random query
        if self.split == "train":
            query_text = random.choice(sample["queries"])
        else:
            query_text = sample["queries"][0]

        hard_neg_texts = sample["hard_negatives"][:4]

        return {
            "visual_features": visual_features,
            "visual_mask": visual_mask,
            "frame_timestamps": frame_timestamps,
            "token_reprs": token_reprs,
            "token_mask": token_mask,
            "segment_pooled": segment_pooled,
            "segment_mask": segment_mask,
            "seg_timestamps": seg_timestamps,
            "query_text": query_text,
            "hard_neg_texts": hard_neg_texts,
            "shot_id": f"{video_id}_{shot_id}",
        }


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    out = {}
    tensor_keys = (
        "visual_features", "visual_mask", "frame_timestamps",
        "token_reprs", "token_mask", "segment_pooled",
        "segment_mask", "seg_timestamps",
    )
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["query_text"] = [b["query_text"] for b in batch]
    out["hard_neg_texts"] = [b["hard_neg_texts"] for b in batch]
    out["shot_id"] = [b["shot_id"] for b in batch]
    return out


def get_dataloader(
    split: str,
    base_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    max_frames: int = 18,
    max_segments: int = 10,
    max_seg_tokens: int = 128,
    visual_feature_subdir: str = "siglip2",
):
    dataset = VietnameseNewsDataset(
        split=split,
        base_dir=base_dir,
        max_frames=max_frames,
        max_segments=max_segments,
        max_seg_tokens=max_seg_tokens,
        visual_feature_subdir=visual_feature_subdir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
