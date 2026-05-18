"""
Extracts representative keyframes from a video clip using scene-change clustering.

Usage:
    python source/keyframe_extractor.py video_output/K01_V001/000.mp4

Output:
    frame_extracted/K01_V001_000/<frame>.jpg  — representative frame images
    time_extract/K01_V001_000.json            — list of {start, end, image}
"""

import cv2
import numpy as np
import os
import json
import subprocess
from pathlib import Path


# ── similarity (same as clip.py) ─────────────────────────────────────────────

def frame_similarity(f1, f2):
    """Weighted combo of HSV histogram, DCT, and pixel cosine similarity."""
    h1 = cv2.calcHist([cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)],
                      [0, 1], None, [32, 32], [0, 180, 0, 256])
    h2 = cv2.calcHist([cv2.cvtColor(f2, cv2.COLOR_BGR2HSV)],
                      [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    s_hsv = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

    def dct_feat(f):
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
        return cv2.dct(g)[:8, :8].flatten()

    d1, d2 = dct_feat(f1), dct_feat(f2)
    s_dct = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-8))

    p1 = cv2.resize(f1, (16, 16)).flatten().astype(np.float32)
    p2 = cv2.resize(f2, (16, 16)).flatten().astype(np.float32)
    s_pix = float(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8))

    return 0.7 * s_hsv + 0.1 * s_dct + 0.2 * s_pix


# ── clustering ────────────────────────────────────────────────────────────────

def cluster_frames(frame_indices, frames_bgr, threshold=0.75):
    """
    Same local-minimum clustering as clip.py.
    Returns list of clusters, each cluster is a list of frame indices.
    """
    clusters = []
    curr_cluster = [frame_indices[0]]
    prev_score = 1.0

    for i in range(1, len(frame_indices)):
        score = frame_similarity(frames_bgr[i - 1], frames_bgr[i])
        is_local_min = (prev_score < threshold) and (score > prev_score)

        if is_local_min:
            clusters.append(curr_cluster)
            curr_cluster = [frame_indices[i]]
        else:
            curr_cluster.append(frame_indices[i])

        prev_score = score

    clusters.append(curr_cluster)
    return clusters


def merge_small_clusters(clusters, min_size=2):
    """
    Clusters with < min_size frames are merged into the previous cluster's
    time range (their frames are discarded but end time extends to their last frame).
    Returns list of (frame_list, end_frame_idx) tuples.
    """
    result = []  # list of (frame_list, actual_end_frame)

    for cluster in clusters:
        if len(cluster) < min_size:
            if result:
                # Extend previous cluster's end time
                prev_frames, _ = result[-1]
                result[-1] = (prev_frames, cluster[-1])
            # If no previous cluster exists, just skip (edge case at start)
        else:
            result.append((cluster, cluster[-1]))

    return result


# ── representative frame selection ───────────────────────────────────────────

def find_representative(cluster_frames_bgr):
    """
    Among all frames in a cluster, returns the index of the frame whose
    average pairwise similarity to all other frames is highest.
    """
    n = len(cluster_frames_bgr)
    if n == 1:
        return 0

    avg_sims = []
    for i in range(n):
        sims = [frame_similarity(cluster_frames_bgr[i], cluster_frames_bgr[j])
                for j in range(n) if j != i]
        avg_sims.append(np.mean(sims))

    return int(np.argmax(avg_sims))


# ── main pipeline ─────────────────────────────────────────────────────────────

def process_video(video_path: str, threshold: float = 0.75):
    """
    Full pipeline: extract → cluster → pick representative → save.

    Args:
        video_path: path to .mp4 file, e.g. "video_output/K01_V001/000.mp4"
        step:       sample every `step` frames (default 25 = 1 s at 25 fps)
        threshold:  similarity threshold for scene-change detection
    """
    video_path = Path(video_path)

    # Derive output names from path  e.g. K01_V001_000
    parts = video_path.parts  # (..., 'K01_V001', '000.mp4')
    folder_name = parts[-2]           # K01_V001
    clip_name   = video_path.stem     # 000
    output_stem = f"{folder_name}_{clip_name}"  # K01_V001_000

    frame_dir  = Path(f"dataset/frame_extracted/{output_stem}")
    json_path  = Path(f"dataset/time_extract/{output_stem}.json")
    frame_dir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # ── read sampled frames ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}  FPS={fps}  Frames={total_frames}  Duration={total_frames/fps:.2f}s")

    sampled_indices = []
    sampled_bgr     = []

    idx = 0
    while idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        sampled_indices.append(idx)
        sampled_bgr.append(frame)
        idx += int(fps)

    cap.release()
    print(f"Sampled {len(sampled_indices)} frames (step={int(fps)})")

    if len(sampled_indices) < 2:
        print("Not enough frames to cluster.")
        return

    # ── cluster ───────────────────────────────────────────────────────────────
    raw_clusters = cluster_frames(sampled_indices, sampled_bgr, threshold)
    print(f"Raw clusters: {len(raw_clusters)}")
    for i, c in enumerate(raw_clusters):
        print(f"  Cluster {i}: {len(c)} frames  [{c[0]} – {c[-1]}]")

    merged = merge_small_clusters(raw_clusters, min_size=2)
    print(f"After merging small clusters: {len(merged)}")

    # ── pick representative + save ────────────────────────────────────────────
    results = []

    for cluster_idx, (frame_list, end_frame) in enumerate(merged):
        # Gather bgr frames for this cluster
        bgr_list = [sampled_bgr[sampled_indices.index(fi)] for fi in frame_list]

        rep_local_idx = find_representative(bgr_list)
        rep_frame_idx = frame_list[rep_local_idx]

        start_sec = round(frame_list[0]  / fps, 3)
        end_sec   = round(end_frame       / fps, 3)

        # Save representative frame
        img_filename = f"{int(rep_frame_idx/fps):03d}.jpg"
        img_path     = frame_dir / img_filename
        cv2.imwrite(str(img_path), bgr_list[rep_local_idx])

        results.append({
            "start": start_sec,
            "end":   end_sec,
            "image": str(img_path).replace("\\", "/"),
        })

        print(f"  [{start_sec:.2f}s - {end_sec:.2f}s]  rep={rep_frame_idx}  -> {img_path}")

    # ── write JSON ────────────────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} clips -> {json_path}")
    return results

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # video_path = "video_output/K01_V001/001.mp4"
    # process_video(video_path, 0.75)
    
    base_dir = Path("dataset/video_output")
    for video_id in range(1, 32):
        folder = base_dir / f"K01_V{video_id:03d}"
        
        for video_file in folder.glob("*.mp4"):
            process_video(str(video_file), 0.75)

