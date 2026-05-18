from keyframe_extractor import *
import shutil


def load_images_from_folder(folder: Path) -> list[tuple[str, np.ndarray]]:
    """Load all jpg images from a folder, return list of (filename, bgr_array)."""
    images = []
    for img_path in sorted(folder.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append((img_path.name, img))
    return images


def move_similar_keyframes(
    true_folder: str | Path,
    keyframes_folder: str | Path,
    threshold: float = 0.9,
):
    """
    For each image in keyframes_folder (e.g. keyframes/K16_V002):
      - compute its max similarity against ALL images in true_folder (e.g. true/K16_V001)
      - if max similarity >= threshold, move it to true/<video_id>/ (create folder if needed)

    Args:
        true_folder:      path to e.g. d:/aic/true/K16_V001
        keyframes_folder: path to e.g. d:/aic/keyframes/K16_V002
        threshold:        minimum similarity to be considered a match (default 0.9)
    """
    true_folder = Path(true_folder)
    keyframes_folder = Path(keyframes_folder)

    # Derive destination folder name from keyframes folder (e.g. K16_V002)
    video_id = keyframes_folder.name  # e.g. K16_V002
    # Destination is true/<video_id>
    dest_folder = true_folder.parent / video_id

    # Load reference images from true/K16_V001
    print(f"Loading reference images from: {true_folder}")
    ref_images = load_images_from_folder(true_folder)
    if not ref_images:
        print(f"No images found in {true_folder}, skipping.")
        return

    # Load candidate images from keyframes/K16_V002
    print(f"Loading candidate images from: {keyframes_folder}")
    candidate_images = load_images_from_folder(keyframes_folder)
    if not candidate_images:
        print(f"No images found in {keyframes_folder}, skipping.")
        return

    moved = []
    for fname, cand_img in candidate_images:
        max_sim = max(
            frame_similarity(cand_img, ref_img)
            for _, ref_img in ref_images
        )
        print(f"  {fname}: max_similarity={max_sim:.4f}")

        if max_sim >= threshold:
            dest_folder.mkdir(parents=True, exist_ok=True)
            src_path = keyframes_folder / fname
            dst_path = dest_folder / fname
            shutil.move(str(src_path), str(dst_path))
            moved.append(fname)
            print(f"    -> MOVED to {dst_path}")

    print(f"\nDone. Moved {len(moved)}/{len(candidate_images)} images to {dest_folder}")
    return moved


if __name__ == "__main__":
    for i in range(12, 33):
        move_similar_keyframes(
            true_folder="d:/aic/true/K16_V001",
            keyframes_folder="d:/aic/keyframes/K16_V{:03d}".format(i),
            threshold=0.86,
        )
